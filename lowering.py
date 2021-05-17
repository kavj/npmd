import itertools
import typing

from collections import defaultdict
from functools import singledispatchmethod

import ir
import reachingcheck
import visitor

"""
This or some other area eventually needs a call stacking mechanic

lowering loops... pre-vectorization

Marking precedes lowering, because lowering generates additional auxiliary stuff that should never be allowed
to break the analyses. It may however obfuscate them, by introducing additional variables, which do not escape
loops or create additional clobbers. As a result, it's best to run analysis and cache this to be used further after
iterator lowering. Alternatively, iterator lowering would need to discard information or be fused into back end 
code generation, which seems too complicated. 

mark liveouts
mark array expressions
mark data refs that require bounds checks
mark memory hazards
lower iterators 
   -- inline flat indexing sentinel
   -- sink loads


validate types / perform inference
check broadcasting

perform vectorization or scalarization
final code gen.. I think...


The advantage of using this over array operations is that it does not require complicated array expressions.
Array expressions and fusion could be added onto this as a separate feature, but this seems helpful in that it can
represent loops with strided access without making dependence analysis or vectorization into a difficult process.

Array arithmetic is better when you have to specialize based on significant work and small arrays, as well as inlined
calculation of stencil calculations.

Varying is top down...
We should revisit type inference and make it as strict as possible, thus requiring
user annotations for anything polymorphic. This is less convenient but transparent. This needs to 
be providable without inlining of annotations, due to the templated nature of this. We can annotate types used
by variable names. This allows better grouping.

Interestingly most of this stuff doesn't require type inference..
I clearly started on that much too early.

"""


def find_live(stmts):
    """
    This is mostly for variable declarations, but it's more annoying than it seems when working with a tree structure,
    since we can have things like

    if cond:
        for .. in ..:
           ...
    else:
       ...

    use(..)

    Since the reaching check is pretty strict, we can check what is declared before this loop.

    """
    rc = reachingcheck.ReachingCheck()
    live, _, _ = rc(stmts)
    return live


def find_nested_loops(stmts):
    nested = []
    for stmt in stmts:
        if isinstance(stmt, ir.IfElse):
            nested.extend(find_nested_loops(stmt.if_branch))
            nested.extend(find_nested_loops(stmt.else_branch))
        elif isinstance(stmt, (ir.ForLoop, ir.WhileLoop)):
            nested.append(stmt)
    return nested


def extract_base(subscr):
    while isinstance(subscr, ir.Subscript):
        subscr = subscr.value
    return subscr


class HeaderInfo:
    """

    this could be folded into the for loop ir node?
    it may be sma0ll enough

    return (indices, constraints)

    indices -- any counting info, with explicit bias, eg. enumerate(array, 10) has bias 10, but we want to return
               the name and bias

    constraints -- array and range constraints

    This way if a variable is bound here, then used for a subscript, we can determine if it must fall within bounds

    """

    # maybe move this out of a constructor

    def __init__(self, header: ir.ForLoop):
        self.indices = {}
        self.constraints = set()

        for target, iterable in header.walk_assignments():
            if isinstance(iterable, ir.Counter):
                # should already be filtered against duplicates
                self.indices[target] = iterable
                if isinstance(iterable.step, ir.IntNode):
                    if iterable.step.value < 0:
                        c = ir.BinOp(iterable.start, iterable.stop, "-")
                    else:
                        c = ir.BinOp(iterable.stop, iterable.start, "-")
                    base_count = ir.BinOp(c, iterable.step, "//")
                    if iterable.stop is None:
                        continue
                    constr = base_count
                    if iterable.step != ir.IntNode(-1):
                        remainder = ir.BinOp(c, iterable.step, "%")
                        test = ir.BinOp(remainder, ir.IntNode(0), "==")
                        tail_iter_count = ir.IfExpr(test, ir.IntNode(0), ir.IntNode(1))
                        constr = ir.BinOp(constr, tail_iter_count, "+")
                    self.constraints.add(constr)
                else:
                    # not much we can do here..
                    # we could disallow arrays with negative array dimensions, which would make
                    # this easier to infer
                    raise ValueError("no support for compiling based on unknown step")
            else:
                # If it's just one, it must always be a constraint
                # since we don't support infinite generators
                self.constraints.add(ir.ShapeRef(iterable, ir.IntNode(0)))

    indices: dict
    constraints: set


class VarInfo:
    """
    should remain on its own..
    this indicates possible hazards in some section of code

    This is meant to be fully constructed prior to querying for info.

    """

    declared_in_outer_scope: bool = False
    read: bool = False
    written: bool = False
    augmented: bool = False  # only count explicit memory updates such as array clobbering
    min_dim_constraint: int = 0  # zero is scalar and not sliceable. There are no zero dim arrays here.

    @property
    def is_array_valued(self):
        return self.min_dim_constraint > 0


class StmtListInfo:
    var_info: typing.Set[VarInfo]
    contains_loops: bool
    contains_exits: bool


class scev:
    def __init__(self, start, step):
        self.start = start
        self.step = step


# could move into an index class with an is_valid_loop_counter parameter
# since this can safely default to false


class VarInfoVisitor(visitor.VisitorBase):

    def __call__(self, entry):
        self.info = defaultdict(ir.NameRef)
        self.visit(entry)
        info = self.info
        self.info = None
        return info

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.ForLoop):
        for target, iterable in node.walk_assignments():
            assert isinstance(target, ir.NameRef)
            self.info[target].written = True
            self.visit(iterable)
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.WhileLoop):
        self.visit(node.test)
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.NameRef):
        self.info[node].read = True

    @visit.register
    def _(self, node: ir.Assign):
        if isinstance(node.target, ir.NameRef):
            self.info[node.target].written = True
        else:
            if isinstance(node.target, ir.Subscript):
                name = extract_base(node.target)
                self.info[name].augmented = True
                self.visit(node.target)
                self.visit(node.value)


def find_valid_loop_counter(header: ir.ForLoop, info: typing.Dict[ir.NameRef, VarInfo]):
    for target, iterable in header.walk_assignments():
        if isinstance(iterable, ir.Counter):
            vi = info[target]
            if (iterable.start == ir.IntNode(0)
                    and iterable.step == ir.IntNode(1)
                    and vi.min_dim_constraint == 0
                    and not vi.written
                    and not vi.augmented):
                return target


def generate_header_assignments(header: ir.ForLoop, loop_index):
    """
    Generate a sequence of assignments that should move to the loop body.
    If any target is to be reused as a loop index, we skip its explicit loop body assignment,
    eg. don't generate an assignment of the form "i = i"

    """
    assigns = []

    # ignore pos for now, this would need to update line number and generate new positional info
    # it really needs a visitor to do this properly though.
    current_pos = header.pos

    for target, iterable in header.walk_assignments():
        if isinstance(iterable, ir.Counter):
            # for simplicity, make new loop index by default
            # This requires far fewer checks..
            expr = loop_index
            if iterable.step != ir.IntNode(1):
                expr = ir.BinOp(iterable.step, expr, "*")
            if iterable.start != ir.IntNode(0):
                expr = ir.BinOp(expr, iterable.start, "+")
            assigns.append(ir.Assign(target, expr, current_pos))
        else:
            assigns.append(ir.Assign(target, ir.Subscript(iterable, loop_index), current_pos))

    return assigns


def lower_iterator_for_loop(header: ir.ForLoop, loop_index: ir.NameRef):
    body = generate_header_assignments(header, loop_index)
    # ignore updating positions for now
    body.extend(header.body)
    hinfo = HeaderInfo(header)
    assigns = [(loop_index, ir.Counter(ir.IntNode(0), ir.MinConstraint(hinfo.constraints), ir.IntNode(1)))]
    return ir.ForLoop(assigns, body, header.pos)


class loop_lower(visitor.TransformBase):

    def make_index_name(self):
        return ir.NameRef(f"{self.prefix}_{next(self.counter)}")

    def __init__(self, prefix):
        self.prefix = prefix
        self.counter = itertools.count()

    def __call__(self, entry):
        repl = self.visit(entry)
        return repl

    def visit(self, node):
        if isinstance(node, ir.ForLoop):
            body = self.visit(node.body)
            index_name = self.make_index_name()
            prologue = generate_header_assignments(node, index_name)
            prologue.extend(body)
            hinfo = HeaderInfo(node)
            if len(hinfo.constraints) == 1:
                constr = hinfo.constraints.pop()
            else:
                constr = ir.MinConstraint(hinfo.constraints)
            assigns = [(index_name, ir.Counter(ir.IntNode(0), constr, ir.IntNode(1)))]
            repl = ir.ForLoop(assigns, prologue, node.pos)
            return repl
        else:
            return super().visit(node)


def scalarize():
    # Todo: add kwargs to ir loop types for loop annotations
    pass
