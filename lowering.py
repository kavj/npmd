import itertools
import operator

from collections import deque
from functools import singledispatchmethod

import ir

from visitor import VisitorBase, walk_all


def extract_name(name):
    return name.name if isinstance(name, ir.NameRef) else name


class name_generator:
    def __init__(self, prefix):
        self.prefix = prefix
        self.gen = itertools.count()

    def make_name(self):
        return f"{self.prefix}_{next(self.gen)}"


class symbol_gen:

    def __init__(self):
        self.names = set()
        self.added = set()
        self.gen = itertools.count()

    def __contains__(self, item):
        if isinstance(item, ir.NameRef):
            item = item.name
        return item in self.names

    def make_unique_name(self, prefix):
        name = f"{prefix}_{next(self.gen)}"
        while name in self.names:
            name = f"{prefix}_{next(self.gen)}"
        self.names.add(name)
        return name


def build_symbols(entry):
    # grabs all names that can be declared at outmermost scope
    names = set()
    if isinstance(entry, ir.Function):
        for arg in entry.args:
            names.add(extract_name(arg))
    for stmt in walk_all(entry):
        if isinstance(stmt, ir.Assign):
            if isinstance(stmt.target, ir.NameRef):
                names.add(extract_name(stmt.target))
        elif isinstance(stmt, ir.ForLoop):
            for target, _ in stmt.walk_assignments():
                names.add(extract_name(target))
    return names


def flatten_branches(header: ir.IfElse):
    """
    return something that matches an elif structure
    the notion is needed to distinguish these from independently taken branches

    """
    if (operator.truth(header.else_branch)
            and len(header.else_branch) == 1
            and isinstance(header.else_branch[0], ir.IfElse)):
        if_stmts = ir.IfElse(header.test, header.if_branch, [], header.pos)
        elif_ = flatten_branches(header.else_branch[0])
        elif_[0].is_elif = True
        elif_.appendleft(if_stmts)
    else:
        if_stmts = deque()
        if_stmts.append(header)
    return if_stmts


def get_leading_dim_bound(index, types):
    return 0 if isinstance(index, ir.Slice) else 1


def is_normalized_bound(iterable, types):
    if isinstance(iterable, ir.Counter):
        return iterable.start == ir.IntNode(0) and iterable.step == ir.IntNode(1)
    array_type = types.get(iterable)
    return not isinstance(array_type.slice, ir.Slice)


def is_dim_reduction(array):
    return isinstance(array, ir.Subscript) and not isinstance(array.slice, ir.Slice)


def is_strided_leading_dim(array, types):
    if is_dim_reduction(array):
        return False
    elif isinstance(array, ir.Subscript):
        s = array.slice
        return s.step != 1 or s.step != ir.IntNode(1)
    else:
        array_type = types.get(array)
        s = array_type.slice
        return s.step != 1 or s.step != ir.IntNode(1)


def get_strided_refs(header: ir.ForLoop, types):
    return {iterable for _, iterable in header.walk_assignments() if is_strided_leading_dim(iterable, types)}


def get_loop_bounds(header, types):
    simple_bounds = set()
    complex_bounds = set()
    for _, iterable in header.walk_assignments():
        if isinstance(iterable, ir.Counter):
            if iterable.start == ir.IntNode(0) and iterable.step == ir.IntNode(1):
                simple_bounds.add(iterable.stop)
            else:
                complex_bounds.add(iterable)
        else:
            if is_strided_leading_dim(iterable, types):
                complex_bounds.add(iterable)
            else:
                array_ref = types.get(iterable)
                simple_bounds.add(array_ref.dims[0])
    return simple_bounds, complex_bounds


def make_min_expr(bounds):
    bounds_iterator = iter(bounds)
    bound = next(bounds_iterator)
    for b in bounds:
        bound = ir.IfExpr(bound, ir.BinOp(bound, b, "<"), b)
    return bound


def generate_escape_branch_expr(complex_bounds):
    """
    This is to avoid loop peeling of many variables, which can become expensive.
    We should be marking these uniform when possible..

    """

    return None


class SimplifyFunction(VisitorBase):

    def __call__(self, entry, types, default_index_type):
        self.types = types
        self.gen = symbol_gen()
        self.default_index_type = default_index_type
        func = self.visit(entry)
        self.types = self.gen = self.default_index_type = None
        return func

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Function):
        body = self.visit(node.body)
        return ir.Function(node.name, node.args, body)

    @visit.register
    def _(self, node: ir.IfElse):
        stmts = flatten_branches(node)
        if len(stmts) == 1:
            return node
        else:
            return stmts

    @visit.register
    def _(self, node: ir.ForLoop):
        simple, complicated = get_loop_bounds(node, self.types)
        loop_index = self.gen.make_unique_name()
        escape_branch_expr = generate_escape_branch_expr(complicated)
        body = self.visit(node.body)
        # generate intro assignments
        body = [ir.IfElse(escape_branch_expr, [ir.Break(node.pos)], body, node.pos)]
        return ir.ForLoop([(loop_index,
                            ir.Counter(ir.IntNode(0), simple, ir.IntNode(1)))],
                          body,
                          node.pos)
