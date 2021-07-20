import ast
import numbers
import sys
import typing

from contextlib import ContextDecorator

import ir
from symbols import create_symbol_tables
from Canonicalize import replace_builtin_call
from lowering import make_loop_interval
from visitor import VisitorBase

binaryops = {ast.Add: "+",
             ast.Sub: "-",
             ast.Mult: "*",
             ast.Div: "/",
             ast.FloorDiv: "//",
             ast.Mod: "%",
             ast.Pow: "**",
             ast.LShift: "<<",
             ast.RShift: ">>",
             ast.BitOr: "|",
             ast.BitXor: "^",
             ast.BitAnd: "&",
             ast.MatMult: "@"}

binary_inplace_ops = {ast.Add: "+=",
                      ast.Sub: "-=",
                      ast.Mult: "*=",
                      ast.Div: "/=",
                      ast.FloorDiv: "//=",
                      ast.Mod: "%=",
                      ast.Pow: "**=",
                      ast.LShift: "<<=",
                      ast.RShift: ">>=",
                      ast.BitOr: "|=",
                      ast.BitXor: "^=",
                      ast.BitAnd: "&=",
                      ast.MatMult: "@="}

unaryops = {ast.UAdd: "+",
            ast.USub: "-",
            ast.Invert: "~",
            ast.Not: "not"}

boolops = {ast.And: "and",
           ast.Or: "or"}

compareops = {ast.Eq: "==",
              ast.NotEq: "!=",
              ast.Lt: "<",
              ast.LtE: "<=",
              ast.Gt: ">",
              ast.GtE: ">=",
              ast.Is: "is",
              ast.IsNot: "isnot",
              ast.In: "in",
              ast.NotIn: "notin"}

supported_builtins = {"iter", "range", "enumerate", "zip", "all", "any", "max", "min", "abs", "pow",
                      "round", "reversed"}


# Prior version maintained a more native assignment structure.
# this is nice for determining what bounds are preserved, but it complicates lowering.


def serialize_iterated_assignments(target, iterable):
    """
    Serializes (target, iterable) pairs by unpacking order
    """
    if (isinstance(target, ir.Tuple)
            and isinstance(iterable, ir.Zip)
            and len(target.elements) == len(iterable.elements)):

        queued = [zip(target.elements, iterable.elements)]

        while queued:
            try:
                t, v = next(queued[-1])
                if (isinstance(t, ir.Tuple)
                        and isinstance(v, ir.Zip)
                        and len(t.elements) == len(v.elements)):
                    queued.append(zip(t.elements, v.elements))
                else:
                    yield t, v
            except StopIteration:
                queued.pop()
    else:
        yield target, iterable


def serialize_assignments(node: ir.Assign):
    """
    This is here to break things like

    (a,b,c,d) = (b,d,c,a)

    into a series of assignments, since these are somewhat rare, and explicitly lowering them
    to shuffles is not worthwhile here.

    """
    if not isinstance(node.target, ir.Tuple) or not isinstance(node.value, ir.Tuple) \
            or len(node.target.elements) != len(node.value.elements):
        return [node]

    queued = [zip(node.target.elements, node.value.elements)]
    while queued:
        try:
            t, v = next(queued[-1])
            if (isinstance(node.target, ir.Tuple)
                    and isinstance(node.value, ir.Tuple)
                    and len(node.target.elements) == len(node.value.elements)):
                queued.append(zip(t.elements, v.elements))
            else:
                yield t, v, node.pos
        except StopIteration:
            queued.pop()


def is_ellipsis(node):
    vi = sys.version_info
    if vi.major != 3 or vi.minor < 8:
        raise NotImplementedError("This only applies to CPython, version 3.8 and higher")
    else:
        # could patch in 7 by checking the deprecated node type.
        if isinstance(node, ast.Constant):
            return node.value == Ellipsis
    return False


def extract_positional_info(node):
    return ir.Position(line_begin=node.lineno,
                       line_end=node.end_lineno,
                       col_begin=node.col_offset,
                       col_end=node.end_col_offset)


def collect_constraint_variables(iters):
    iterable_constraints = set()
    range_constraints = set()

    for iterable in iters:
        if isinstance(iterable, ir.Counter):
            if iterable.stop is not None:
                #  keep step since we may not know sign
                range_constraints.add(iterable)
        else:
            iterable_constraints.add(iterable)
    return iterable_constraints, range_constraints


class FlowContext(ContextDecorator):
    """
    Start of a loop context manager
    """

    def __init__(self, builder, header):
        assert isinstance(header, (ast.For, ast.If, ast.While, ast.FunctionDef))
        self.builder = builder
        self.header = header
        self.enters_loop = isinstance(header, (ast.For, ast.While))
        self.body = []

    def __enter__(self):
        self.builder.body, self.body = self.body, self.builder.body
        if self.enters_loop:
            self.builder.enclosing_loop, self.header = self.header, self.builder.enclosing_loop

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.builder.body, self.body = self.body, self.builder.body
        if self.enters_loop:
            self.builder.enclosing_loop, self.header = self.header, self.builder.enclosing_loop
        if exc_type:
            raise


class AnnotationCollector(VisitorBase):
    """
    This tries to parse anything the grammar can handle.
    Note: Annotations are no longer used, since they make templating awkward.

    """

    def __call__(self, node):
        return self.visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        value = self.visit(node.value)
        if isinstance(value, ir.AttributeRef):
            value = ir.AttributeRef(value.value, value.attr + node.attr)
        else:
            value = ir.AttributeRef(value, node.attr)
        return value

    def visit_Constant(self, node: ast.Constant):
        if is_ellipsis(node.value):
            raise NotImplementedError("Ellipsis is not yet supported.")
        if isinstance(node.value, str):
            return ir.StringNode(node.value)
        elif isinstance(node.value, numbers.Integral):
            return ir.IntNode(node.value)
        elif isinstance(node.value, numbers.Real):
            return ir.FloatNode(node.value)
        else:
            raise TypeError("unrecognized constant type")

    def visit_Name(self, node: ast.Name):
        return ir.NameRef(node.id)

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.value, ast.Subscript):
            raise NotImplementedError("Annotations aren't supported with nested subscripts")
        target = self.visit(node.value)
        s = self.visit(node.slice)
        if isinstance(target, ir.Subscript):
            # annotations can't have consecutive subscripts like name[...][...]
            raise ValueError
        return ir.Subscript(target, s)

    def visit_Index(self, node: ast.Index):
        return self.visit(node.value)

    def visit_Slice(self, node: ast.Slice):
        lower = self.visit(node.lower) if node.lower is not None else None
        upper = self.visit(node.upper) if node.upper is not None else None
        step = self.visit(node.step) if node.step is not None else None
        return ir.Slice(lower, upper, step)

    def generic_visit(self, node):
        raise NotImplementedError(f"{type(node)} is not supported here for type annotations.")


class TreeBuilder(ast.NodeVisitor):

    def __call__(self, entry: ast.FunctionDef, syms):
        self.enclosing_loop = None
        self.entry = entry
        self.syms = syms
        self.body = []
        func = self.visit(entry)
        assert self.enclosing_loop is None
        return func

    def make_flow_region(self, header):
        if not isinstance(header, (ast.For, ast.While, ast.If)):
            msg = f"{header} is not a valid control flow entry point."
            raise ValueError(msg)
        return FlowContext(self, header)

    def visit_Attribute(self, node: ast.Attribute) -> ir.AttributeRef:
        value = self.visit(node.value)
        if node.attr == "shape":
            value = ir.ShapeRef(value)
        elif isinstance(value, ir.AttributeRef):
            value = ir.AttributeRef(value.value, value.attr + node.attr)
        else:
            value = ir.AttributeRef(value, node.attr)
        return value

    def visit_Constant(self, node: ast.Constant) -> ir.Constant:
        if is_ellipsis(node.value):
            raise TypeError
        if isinstance(node.value, str):
            return ir.StringNode(node.value)
        elif isinstance(node.value, numbers.Integral):
            return ir.IntNode(node.value)
        elif isinstance(node.value, numbers.Real):
            return ir.FloatNode(node.value)
        else:
            raise TypeError("unrecognized constant type")

    def visit_Tuple(self, node: ast.Tuple) -> ir.Tuple:
        # refactor seems to have gone wrong here..
        # version = sys.version_info
        # if (3, 9) <= (version.major, version.minor):
        # 3.9 removes ext_slice in favor of a tuple of slices
        # need to check a lot of cases for this
        #    raise NotImplementedError
        elts = tuple(self.visit(elt) for elt in node.elts)
        return ir.Tuple(elts)

    def visit_Name(self, node: ast.Name):
        return ir.NameRef(node.id)

    def visit_Expr(self, node: ast.Expr):
        # single expression statement
        expr = self.visit(node.value)
        pos = extract_positional_info(node)
        self.body.append(ir.SingleExpr(expr, pos))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ir.UnaryOp:
        op = unaryops.get(type(node.op))
        operand = self.visit(node.operand)
        return ir.UnaryOp(operand, op) if op != "+" else operand

    def visit_BinOp(self, node: ast.BinOp) -> ir.BinOp:
        op = binaryops.get(type(node.op))
        left = self.visit(node.left)
        right = self.visit(node.right)
        return ir.BinOp(left, right, op)

    def visit_BoolOp(self, node: ast.BoolOp) -> ir.BoolOp:
        op = boolops[node.op]
        operands = tuple(self.visit(value) for value in node.values)
        return ir.BoolOp(operands, op)

    def visit_Compare(self, node: ast.Compare) -> typing.Union[ir.BinOp, ir.BoolOp]:
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        op = compareops[type(node.ops[0])]
        initial_compare = ir.BinOp(left, right, op)
        if len(node.ops) == 1:
            return initial_compare
        compares = [initial_compare]
        for index, ast_op in enumerate(node.ops[1:], 1):
            # expressions are immutable, so we can safely reuse them
            left = right
            right = self.visit(node.comparators[index])
            op = compareops[type(ast_op)]
            compares.append(ir.BinOp(left, right, op))
        return ir.BoolOp(tuple(compares), "and")

    def visit_Call(self, node: ast.Call) -> typing.Union[ir.Expression, ir.NameRef]:
        if isinstance(node.func, ast.Name):
            funcname = node.func.id
        else:
            funcname = self.visit(node.func)
        args = tuple(self.visit(arg) for arg in node.args)
        keywords = tuple((kw.arg, self.visit(kw.value)) for kw in node.keywords)
        # replace call should handle folding of casts
        call_ = ir.Call(funcname, args, keywords)
        # Todo: need a way to identify array creation
        func = replace_builtin_call(call_)
        return func

    def visit_IfExp(self, node: ast.IfExp) -> ir.IfExpr:
        test = self.visit(node.test)
        on_true = self.visit(node.body)
        on_false = self.visit(node.orelse)
        return ir.IfExpr(test, on_true, on_false)

    def visit_Subscript(self, node: ast.Subscript) -> ir.Subscript:
        target = self.visit(node.value)
        s = self.visit(node.slice)
        value = ir.Subscript(target, s)
        return value

    def visit_Index(self, node: ast.Index) -> typing.Union[ir.Expression, ir.NameRef, ir.Constant]:
        return self.visit(node.value)

    def visit_Slice(self, node: ast.Slice) -> ir.Slice:
        lower = self.visit(node.lower) if node.lower is not None else None
        upper = self.visit(node.upper) if node.upper is not None else None
        step = self.visit(node.step) if node.step is not None else None
        return ir.Slice(lower, upper, step)

    def visit_ExtSlice(self, node: ast.ExtSlice):
        # This is probably never going to be supported, because it requires inlining
        # a large number of calculations in ways that may sometimes hamper performance.
        raise TypeError("Extended slices are currently unsupported. This supports single"
                        "slices per dimension")

    def visit_AugAssign(self, node: ast.AugAssign):
        target = self.visit(node.target)
        operand = self.visit(node.value)
        op = binary_inplace_ops.get(type(node.op))
        pos = extract_positional_info(node)
        assign = ir.Assign(target, ir.BinOp(target, operand, op), pos)
        self.body.append(assign)

    def visit_Assign(self, node: ast.Assign):
        # first convert locally to internal IR
        value = self.visit(node.value)
        pos = extract_positional_info(node)
        for target in node.targets:
            # break cascaded assignments into multiple assignments
            target = self.visit(target)
            initial = ir.Assign(target, value, pos)
            for subtarget, subvalue, pos in serialize_assignments(initial):
                if isinstance(subtarget, ir.Tuple) or isinstance(subvalue, ir.Tuple):
                    raise TypeError(f"unable to determine that assignment at line {pos.line_begin} is fully unpackable.")
                subassign = ir.Assign(subtarget, subvalue, pos)
                self.body.append(subassign)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        target = self.visit(node.target)
        value = self.visit(node.value)
        pos = extract_positional_info(node)
        # We ignore these, because they don't scale well
        # based on parametric constraints
        assign = ir.Assign(target, value, pos)
        self.body.append(assign)

    def visit_Pass(self, node: ast.Pass):
        pos = extract_positional_info(node)
        stmt = ir.Pass(pos)
        return stmt

    def visit_If(self, node: ast.If):
        pos = extract_positional_info(node)
        compare = self.visit(node.test)
        with self.make_flow_region(node):
            for stmt in node.body:
                self.visit(stmt)
            on_true = self.body
        with self.make_flow_region(node):
            for stmt in node.orelse:
                self.visit(stmt)
            on_false = self.body
        ifstat = ir.IfElse(compare, on_true, on_false, pos)
        self.body.append(ifstat)

    def visit_For(self, node: ast.For):
        if node.orelse:
            raise ValueError("or else clause not supported for for statements")
        iter_node = self.visit(node.iter)
        target_node = self.visit(node.target)
        pos = extract_positional_info(node)
        targets = []
        iterables = []
        prefix = None
        # Construct the actual loop header, using a single induction variable.
        with self.make_flow_region(node):
            for target, iterable in serialize_iterated_assignments(target_node, iter_node):
                if isinstance(target, ir.Tuple) or isinstance(iterable, ir.Zip):
                    msg = f"Unable to fully unpack loop, line: {node.lineno}"
                    raise ValueError(msg)
                elif isinstance(target, ir.Subscript):
                    # Catch this early, because otherwise it manifests in weird ways.
                    # These can render it impossible to fully group memory loads at the beginning of a loop.
                    msg = f"Subscripts are not supported as for loop targets, line: {pos.line_begin}"
                    raise ValueError(msg)
                if prefix is None and isinstance(iterable, ir.Counter):
                    if iterable.stop is None:
                        # first enumerate encountered
                        loop_index_prefix = target
                targets.append(target)
                iterables.append(iterable)
            conflicts = set(targets).intersection(iterables)
            if conflicts:
                conflict_names = ", ".join(c for c in conflicts)
                msg = f"{conflict_names} appear in both the target an iterable sequences of a for loop, " \
                      f"line {pos.line_begin}. This is not supported."
                raise ValueError(msg)
            # make a unique loop index name, which cannot escape the current scope
            if prefix is None:
                prefix = "i"
            loop_index = self.syms.add_var(prefix, self.syms.default_int, is_added=True)
            loop_counter = make_loop_interval(targets, iterables, self.syms, loop_index)
            for stmt in node.body:
                self.visit(stmt)
            # This can still be an ill formed loop, eg one with an unreachable latch.
            # These are not supported, because they look weird after transformation to a legal one
            # and raising an error is less mysterious. They should be be caught during tree validation checks.
            loop = ir.ForLoop(loop_index, loop_counter, self.body, pos)
        loop_bound = ir.Assign(loop_index, loop_counter, pos)
        # append loop bound calculations outside loop
        self.body.append(loop_bound)
        self.body.append(loop)

    def visit_While(self, node: ast.While):
        if node.orelse:
            raise ValueError("or else clause not supported for for statements")
        test = self.visit(node.test)
        pos = extract_positional_info(node)
        with self.make_flow_region(node):
            for stmt in node.body:
                self.visit(stmt)
            loop = ir.WhileLoop(test, self.body, pos)
        self.body.append(loop)

    def visit_Break(self, node: ast.Break):
        if self.enclosing_loop is None:
            raise ValueError("Break encountered outside of loop.")
        pos = extract_positional_info(node)
        stmt = ir.Break(pos)
        self.body.append(stmt)

    def visit_Continue(self, node: ast.Continue):
        if self.enclosing_loop is None:
            raise ValueError("Continue encountered outside of loop.")
        pos = extract_positional_info(node)
        stmt = ir.Continue(pos)
        self.body.append(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # can't check just self.blocks, since we automatically 
        # instantiate entry and exit
        if node is not self.entry:
            raise RuntimeError(f"Nested scopes are unsupported. line: {node.lineno}")
        name = node.name
        # Todo: warn about unsupported argument features
        params = [ir.NameRef(arg.arg) for arg in node.args.args]
        for stmt in node.body:
            self.visit(stmt)
        func = ir.Function(name, params, self.body)
        return func

    def visit_Return(self, node: ast.Return):
        pos = extract_positional_info(node)
        value = self.visit(node.value) if node.value is not None else None
        stmt = ir.Return(value, pos)
        self.body.append(stmt)

    def generic_visit(self, node):
        raise NotImplementedError(f"{type(node)} is unsupported")


def build_module_ir(src, filename, types):
    """
    Module level point of entry for IR construction.

    Parameters
    ----------

    src: str
        Source code for the corresponding module

    filename:
        File path we used to extract source. This is used for error reporting.

    types:
        Map of function parameters and local variables to numpy or python numeric types.
   
    """
    # Type info helps annotate things that
    # are not reasonable to type via annotations
    # It's not optimal, but this isn't necessarily
    # meant to be user facing.
    symbols = create_symbol_tables(src, filename, types)
    tree = ast.parse(src)
    tree = ast.fix_missing_locations(tree)

    # scan module for functions

    funcs = []
    imports = []
    build_func_ir = TreeBuilder()
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            syms = symbols[stmt.name]
            func = build_func_ir(stmt, syms)
            funcs.append(func)
        elif isinstance(stmt, ast.Import):
            pos = extract_positional_info(stmt)
            for imp in stmt.names:
                imports.append(ir.ModImport(imp.name, imp.asname, pos))
        elif isinstance(stmt, ast.ImportFrom):
            mod = stmt.module
            pos = extract_positional_info(stmt)
            for name in stmt.names:
                imports.append(ir.NameImport(mod, name.name, name.asname, pos))
        else:
            raise ValueError("Unsupported")

    return ir.Module(funcs, imports)
