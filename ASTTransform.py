import ast
import numbers
import sys
import typing

import ir
from replace_call import replace_builtin_call

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


def serialize_iterated_assignments(target, iterable) -> typing.List[typing.Tuple[ir.Targetable, ir.ValueRef]]:
    """
    Serializes (target, iterable) pairs by unpacking order
    """
    if (isinstance(target, ir.Tuple)
            and isinstance(iterable, ir.Zip)
            and len(target.elements) == len(iterable.elements)):

        queued = [zip(target.elements, iterable.elements)]
        assigns = []

        while queued:
            try:
                t, v = next(queued[-1])
                if (isinstance(t, ir.Tuple)
                        and isinstance(v, ir.Zip)
                        and len(t.elements) == len(v.elements)):
                    queued.append(zip(t.elements, v.elements))
                else:
                    assigns.append((t, v))
            except StopIteration:
                queued.pop()
    else:
        assigns = [(target, iterable)]

    return assigns


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
    assigns = []
    while queued:
        try:
            t, v = next(queued[-1])
            if (isinstance(node.target, ir.Tuple)
                    and isinstance(node.value, ir.Tuple)
                    and len(node.target.elements) == len(node.value.elements)):
                queued.append(zip(t.elements, v.elements))
            else:
                assigns.append(ir.Assign(t, v, node.pos))
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


def build_func_params(node: ast.FunctionDef):
    typer = AnnotationCollector()
    params = []
    default_value_count = len(node.args.defaults)
    arg_count = len(node.args.args)
    for i in range(arg_count):
        arg = node.args.args[arg_count - i - 1]
        if i < default_value_count:
            dv = node.args.defaults[default_value_count - i - 1]
        else:
            dv = None
        annot = typer(arg.annotation) if arg.annotation else None
        commt = typer(arg.type_comment) if arg.type_comment else None
        params.append(ir.Argument(ir.NameRef(arg.arg), annot, commt, dv))
    params.reverse()
    return params


def extract_positional_info(node):
    return ir.Position(line_begin=node.lineno,
                       line_end=node.end_lineno,
                       col_begin=node.col_offset,
                       col_end=node.end_col_offset)


class AnnotationCollector(ast.NodeVisitor):
    """
    This tries to parse anything the grammar can handle.

    """

    def __call__(self, node):
        return self.visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        value = self.visit(node.value)
        if isinstance(value, ir.AttributeRef):
            value = ir.AttributeRef(value.value, value.attr + node.attr)
        else:
            value = ir.AttributeRef(value, (node.attr,))
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

    def __call__(self, entry: ast.FunctionDef):
        self.loops = []
        self.nested_scopes = []
        self.entry = entry
        func = self.visit(entry)
        assert (not self.loops)
        return func, self.nested_scopes

    def visit_body(self, stmts: list):
        body = []
        for stmt in stmts:
            ir_stmt = self.visit(stmt)
            if isinstance(ir_stmt, list):
                body.extend(ir_stmt)
            else:
                body.append(ir_stmt)
        return body

    def visit_Attribute(self, node: ast.Attribute) -> ir.AttributeRef:
        value = self.visit(node.value)
        if node.attr == "shape":
            value = ir.ShapeRef(value)
        elif isinstance(value, ir.AttributeRef):
            value = ir.AttributeRef(value.value, value.attr + node.attr)
        else:
            value = ir.AttributeRef(value, (node.attr,))
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
        version = sys.version_info
        if (3, 9) <= (version.major, version.minor):
            # 3.9 removes ext_slice in favor of a tuple of slices
            # need to check a lot of cases for this            
            raise NotImplementedError
        elts = tuple(self.visit(elt) for elt in node.elts)
        return ir.Tuple(elts)

    def visit_Name(self, node: ast.Name):
        return ir.NameRef(node.id)

    def visit_Expr(self, node: ast.Expr) -> ir.SingleExpr:
        expr = self.visit(node.value)
        pos = extract_positional_info(node)
        return ir.SingleExpr(expr, pos)

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

    def visit_ExtSlice(self, node: ast.ExtSlice) -> ir.Unsupported:
        # This is probably never going to be supported, because it requires inlining
        # a large number of calculations in ways that may sometimes hamper performance.
        return ir.Unsupported("ast.ExtSlice", msg="Extended slices are currently unsupported. This supports single"
                                                  "slices per dimension")

    def visit_AugAssign(self, node: ast.AugAssign) -> ir.Assign:
        target = self.visit(node.target)
        operand = self.visit(node.value)
        op = binary_inplace_ops.get(type(node.op))
        pos = extract_positional_info(node)
        return ir.Assign(target, ir.BinOp(target, operand, op), pos)

    def visit_Assign(self, node: ast.Assign) -> typing.List[ir.Assign]:
        value = self.visit(node.value)
        targets = [self.visit(t) for t in node.targets]
        pos = extract_positional_info(node)
        # lower a = b = c into a = c; b = c
        # (Notice Python uses cascading rules that differ from C, which would serialize
        #  a = b = c into b = c; a = b;)
        assigns = [ir.Assign(target, value, pos) for target in targets]
        # Serialize the case of (a,b,c,d) = (e,f,g,h)
        return assigns

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ir.Assign:
        target = self.visit(node.target)
        value = self.visit(node.value)
        pos = extract_positional_info(node)
        # This should be parsed here later on.
        annot = node.annotation
        return ir.Assign(target, value, pos, annot)

    def visit_Pass(self, node: ast.Pass) -> ir.Pass:
        pos = extract_positional_info(node)
        return ir.Pass(pos)

    def visit_If(self, node: ast.If) -> ir.IfElse:
        pos = extract_positional_info(node)
        compare = self.visit(node.test)
        on_true = self.visit_body(node.body)
        on_false = self.visit_body(node.orelse)
        ifstat = ir.IfElse(compare, on_true, on_false, pos)
        return ifstat

    def visit_With(self, node: ast.With):
        pass

    def visit_For(self, node: ast.For) -> ir.ForLoop:
        if node.orelse:
            raise ValueError("or else clause not supported for for statements")
        it = self.visit(node.iter)
        target = self.visit(node.target)
        pos = extract_positional_info(node)
        assigns = serialize_iterated_assignments(target, it)
        self.loops.append(node)
        body = self.visit_body(node.body)
        self.loops.pop()
        return ir.ForLoop(assigns, body, pos)

    def visit_While(self, node: ast.While) -> ir.WhileLoop:
        if node.orelse:
            raise ValueError("or else clause not supported for for statements")
        compare = self.visit(node.test)
        pos = extract_positional_info(node)
        self.loops.append(node)
        body = self.visit_body(node.body)
        self.loops.pop()
        return ir.WhileLoop(compare, body, pos)

    def visit_Break(self, node: ast.Break):
        if self.loops:
            pos = extract_positional_info(node)
            stmt = ir.Break(pos)
            return stmt
        else:
            raise ValueError("Break encountered outside of loop.")

    def visit_Continue(self, node: ast.Continue) -> ir.Continue:
        if self.loops:
            # Add an edge without explicitly terminating the block.
            pos = extract_positional_info(node)
            return ir.Continue(pos)
        else:
            raise ValueError("Continue encountered outside of loop.")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ir.Function:
        # can't check just self.blocks, since we automatically 
        # instantiate entry and exit

        if node is self.entry:
            name = node.name
            args = build_func_params(node)
            body = self.visit_body(node.body)
            return ir.Function(name, args, body, [], [])
        else:
            self.nested_scopes.append(node)

    def visit_Return(self, node: ast.Return) -> ir.Return:
        pos = extract_positional_info(node)
        value = self.visit(node.value) if node.value is not None else None
        return ir.Return(value, pos)

    def generic_visit(self, node):
        raise NotImplementedError(f"{type(node)} is unsupported")


def build_module_ir(src):
    """
    Module level point of entry for IR construction.

    Parameters
    ----------

    src: str
        Source code for the corresponding module
   
    """
    # Type info helps annotate things that
    # are not reasonable to type via annotations
    # It's not optimal, but this isn't necessarily
    # meant to be user facing.
    tree = ast.parse(src)
    tree = ast.fix_missing_locations(tree)

    # scan module for functions

    funcs = []
    imports = []
    unsupported = []
    build_func_ir = TreeBuilder()
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            func, nested = build_func_ir(stmt)
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
            unsupported.append(stmt)

    if unsupported:
        # This is either a class, control flow, 
        # or data flow at module scope.
        # It should output formatted errors.
        raise ValueError("Unsupported")

    return ir.Module(funcs, imports)
