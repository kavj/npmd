from functools import singledispatchmethod
from contextlib import contextmanager

import ir


#     https://docs.python.org/3/reference/expressions.html#operator-precedence

binop_ordering = {"**": 1, "*": 3, "@": 3, "/": 3, "//": 3, "%": 3, "+": 4, "-": 4, "<<": 5, ">>": 5, "&": 6,
                  "^": 7, "|": 8, "in": 9, "not in": 9, "<": 9, "<=": 9, ">": 9, ">=": 9, "!=": 9,
                  "==": 9}

# Todo: Given the boolean refactoring, not should probably derive from BoolOp, similar to TRUTH.

# Note, python docs don't specify truth precedence, but it should match logical "not"


def check_precedence(node):
    if isinstance(node, ir.ValueRef) and not isinstance(node, ir.Expression):
        prec = 0
    elif isinstance(node, (ir.Subscript, ir.Call, ir.Max, ir.Min, ir.Reversed)):
        prec = 0
    elif isinstance(node, ir.BinOp):
        op = node.op
        prec = binop_ordering[op]
    elif isinstance(node, ir.UnaryOp):
        prec = 2
    elif isinstance(node, (ir.TRUTH, ir.NOT)):
        prec = 10
    elif isinstance(node, (ir.AND, ir.OR)):
        prec = 11
    elif isinstance(node, ir.OR):
        prec = 12
    elif isinstance(node, ir.Ternary):
        prec = 13
    else:
        msg = f"Unable to evaluate precedence for node {node} of type {type(node)}."
        raise ValueError(msg)
    return prec


class pretty_formatter:
    """
    The pretty printer is intended as a way to show the state of the IR in a way that resembles a
    typical source representation.


    """

    def __init__(self):
        self.add_parentheses = False

    def __call__(self, node):
        assert self.add_parentheses is False
        expr = self.visit(node)
        return expr

    @singledispatchmethod
    def visit(self, node):
        msg = f"No method to format node: {node}."
        raise NotImplementedError(msg)

    @visit.register
    def _(self, node: ir.Length):
        expr = self.visit(node.value)
        return f"len({expr})"

    @visit.register
    def _(self, node: ir.Max):
        args = ", ".join(self.visit(arg) for arg in node.values)
        return f"max({args})"

    @visit.register
    def _(self, node: ir.Min):
        args = ", ".join(self.visit(arg) for arg in node.values)
        return f"min({args})"

    @visit.register
    def _(self, node: ir.Ternary):
        test = self.visit(node.test)
        if isinstance(node.test, ir.Ternary):
            test = f"({test})"
        if_expr = self.visit(node.if_expr)
        if isinstance(node.if_expr, ir.Ternary):
            if_expr = f"({if_expr})"
        else_expr = self.visit(node.else_expr)
        if isinstance(node.else_expr, ir.Ternary):
            else_expr = f"({else_expr})"
        expr = f"{if_expr} if {test} else {else_expr}"
        return expr

    @visit.register
    def _(self, node: ir.BoolConst):
        return str(node.value)

    @visit.register
    def _(self, node: ir.IntConst):
        return str(node.value)

    @visit.register
    def _(self, node: ir.FloatConst):
        return str(node.value)

    @visit.register
    def _(self, node: ir.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op
        if not node.in_place:
            node_order = check_precedence(node)
            left_order = check_precedence(node.left)
            right_order = check_precedence(node.right)
            if node_order < left_order:
                left = f"({left})"
            if node_order < right_order:
                right = f"({right})"
        expr = f"{left} {op} {right}"
        return expr

    @visit.register
    def _(self, node: ir.AND):
        node_order = check_precedence(node)
        operands = []
        for operand in node.operands:
            formatted = self.visit(operand)
            order = check_precedence(operand)
            if node_order < order:
                formatted = f"({formatted})"
            operands.append(formatted)
        expr = " and ".join(operand for operand in operands)
        return expr

    @visit.register
    def _(self, node: ir.OR):
        node_order = check_precedence(node)
        operands = []
        for operand in node.operands:
            formatted = self.visit(operand)
            order = check_precedence(operand)
            if node_order < order:
                formatted = f"({formatted})"
            operands.append(formatted)
        expr = " or ".join(operand for operand in operands)
        return expr

    @visit.register
    def _(self, node: ir.NameRef):
        expr = node.name
        return expr

    @visit.register
    def _(self, node: ir.Call):
        func_name = self.visit(node.func)
        args = ", ".join(self.visit(arg) for arg in node.args)
        func = f"{func_name}({args})"
        return func

    @visit.register
    def _(self, node: ir.Reversed):
        return f"reversed({self.visit(node.iterable)})"

    @visit.register
    def _(self, node: ir.Subscript):
        s = f"{self.visit(node.value)}[{self.visit(node.slice)}]"
        return s

    @visit.register
    def _(self, node: ir.AffineSeq):
        # Initial input source may not be easily disernible,
        # print as range
        start = self.visit(node.start)
        stop = self.visit(node.stop) if node.stop is not None else f"None"
        step = self.visit(node.step)
        return f"range({start}, {stop}, {step})"

    @visit.register
    def _(self, node: ir.Tuple):
        elements = []
        for e in node.elements:
            expr = self.visit(e)
            # parenthesize nested tuples, leave everything else
            if isinstance(e, ir.Tuple):
                expr = f"({expr})"
            elements.append(expr)
        s = ", ".join(e for e in elements)
        return s

    @visit.register
    def _(self, node: ir.UnaryOp):
        op = node.op
        operand = self.visit(node.operand)
        if check_precedence(node) < check_precedence(node.operand):
            expr = f"{op}({operand})"
        else:
            expr = f"{op}{operand}"
        return expr

    @visit.register
    def _(self, node: ir.Zip):
        if len(node.elements) == 2:
            first, second = node.elements
            if isinstance(first, ir.AffineSeq):
                if first.stop is None:
                    # This is implicitly convertible to an enumerate expression.
                    inner_expr = self.visit(second)
                    if first.start == ir.Zero:
                        # ignore default value
                        expr = f"enumerate({inner_expr})"
                    else:
                        start = self.visit(first.start)
                        expr = f"enumerate({inner_expr}, {start})"
                    return expr
        exprs = []
        for elem in node.elements:
            formatted = self.visit(elem)
            if isinstance(elem, ir.Tuple):
                # This nesting is unsupported elsewhere, but this
                # would be a confusing place to throw an error.
                formatted = f"({formatted})"
            exprs.append(formatted)
        # handle case of enumerate
        expr = ", ".join(e for e in exprs)
        expr = f"zip({expr})"
        return expr


class printtree:
    """
    Pretty prints tree. 
    Inserts pass on empty if statements or for/while loops.

    """

    def __init__(self, single_indent="    "):
        self.indent = ""
        self._increment = len(single_indent)
        self._single_indent = single_indent
        self.format = pretty_formatter()

    def __call__(self, tree):
        assert self.indent == ""
        self.visit(tree)

    @contextmanager
    def indented(self):
        self.indent = f"{self.indent}{self._single_indent}"
        yield
        self.indent = self.indent[:-self._increment]

    def print_line(self, as_str):
        line = f"{self.indent}{as_str}"
        print(line)

    @singledispatchmethod
    def visit(self):
        raise NotImplementedError

    @visit.register
    def _(self, node: ir.ModImport):
        module = self.format(node.module)
        module_alias = self.format(node.as_name)
        if module == module_alias:
            as_str = f"import {module}"
        else:
            as_str = f"import {module} as {module_alias}"
        self.print_line(as_str)

    @visit.register
    def _(self, node: ir.NameImport):
        module = self.visit(node.module)
        imported_name = self.visit(node.name)
        import_alias = self.visit(node.as_name)
        if imported_name == import_alias:
            as_str = f"from {module} import {imported_name}"
        else:
            as_str = f"from {module} import {imported_name} as {import_alias}"
        self.print_line(as_str)

    @visit.register
    def _(self, node: ir.Return):
        if node.value is None:
            self.print_line("return")
        else:
            expr = self.format(node.value)
            stmt = f"return {expr}"
            self.print_line(stmt)

    @visit.register
    def _(self, node: ir.Module):
        for f in node.funcs:
            print('\n')
            self.visit(f)
            print('\n')

    @visit.register
    def _(self, node: ir.Function):
        name = node.name
        args = ", ".join(self.format(arg) for arg in node.args)
        header = f"{name}({args}):"
        self.print_line(header)
        with self.indented():
            self.visit(node.body)

    @visit.register
    def _(self, node: list):
        for stmt in node:
            self.visit(stmt)

    @visit.register
    def _(self, node: ir.ForLoop):
        target = self.format(node.target)
        iterable = self.format(node.iterable)
        stmt = f"for {target} in {iterable}:"
        self.print_line(stmt)
        with self.indented():
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.WhileLoop):
        test = self.format(node.test)
        stmt = f"while {test}:"
        self.print_line(stmt)
        with self.indented():
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.IfElse):
        test = self.format(node.test)
        stmt = f"if {test}:"
        self.print_line(stmt)
        with self.indented():
            self.visit(node.if_branch)
        if node.else_branch:
            self.print_line("else:")
            with self.indented():
                self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.Assign):
        if node.in_place:
            stmt = self.format(node.value)
        else:
            stmt = f"{self.format(node.target)} = {self.format(node.value)}"
        self.print_line(stmt)

    @visit.register
    def _(self, node: ir.Continue):
        self.print_line("continue")

    @visit.register
    def _(self, node: ir.Break):
        self.print_line("break")

    @visit.register
    def _(self, node: ir.SingleExpr):
        expr = self.format(node.expr)
        self.print_line(expr)
