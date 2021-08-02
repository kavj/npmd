import textwrap

from functools import singledispatchmethod
from contextlib import contextmanager

import ir

from visitor import ExpressionVisitor


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

    @contextmanager
    def parenthesized(self):
        stashed = self.add_parentheses
        self.add_parentheses = True
        yield
        self.add_parentheses = stashed

    @contextmanager
    def no_parentheses(self):
        stashed = self.add_parentheses
        self.add_parentheses = False
        yield
        self.add_parentheses = stashed

    @singledispatchmethod
    def visit(self, node):
        msg = f"No method to format node: {node}."
        raise NotImplementedError(msg)

    @visit.register
    def _(self, node: ir.Length):
        with self.no_parentheses():
             expr = self.visit(node.value)
        return f"len({expr})"

    @visit.register
    def _(self, node: ir.Max):
        with self.no_parentheses():
            args = ", ".join(self.visit(arg) for arg in node.values)
        return f"max({args})"

    @visit.register
    def _(self, node: ir.Min):
        with self.no_parentheses():
            args = ", ".join(self.visit(arg) for arg in node.values)
        return f"min({args})"

    @visit.register
    def _(self, node: ir.Ternary):
        with self.no_parentheses():
            expr = f"{self.visit(node.if_expr)} if {self.visit(node.test)} else {self.visit(node.else_expr)}"
        if self.add_parentheses:
            expr = f"({expr})"
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
        expr = f"{left} {node.op} {right}"
        if self.add_parentheses:
            expr = f"({expr})"
        return expr

    @visit.register
    def _(self, node: ir.AND):
        with self.parenthesized():
            expr = " and ".join(self.visit(operand) for operand in node.operands)
        if self.add_parentheses:
            expr = f"({expr})"
        return expr

    @visit.register
    def _(self, node: ir.OR):
        with self.parenthesized():
            expr = " or ".join(self.visit(operand) for operand in node.operands)
        if self.add_parentheses:
            expr = f"({expr})"
        return expr

    @visit.register
    def _(self, node: ir.NameRef):
        expr = node.name
        return expr

    @visit.register
    def _(self, node: ir.Call):
        func_name = self.visit(node.func)
        if node.args:
            args = ", ".join(self.visit(arg) for arg in node.args)
            func = f"{func_name}({args})"
        else:
            func = f"{func_name}()"
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
        s = ", ".join(self.visit(e) for e in node.subexprs)
        return s

    @visit.register
    def _(self, node: ir.UnaryOp):
        op = node.op
        operand = node.operand
        if op == "not":
            expr = f"not {self.visit(operand)}"
        else:
            expr = f"{op}{self.visit(operand)}"
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
        if node.asname:
            return f"import {node.mod} as {node.asname}"
        else:
            return f"import {node.mod}"

    @visit.register
    def _(self, node: ir.NameImport):
        if node.asname:
            return f"from {node.mod} import {node.name}"
        else:
            return f"from {node.mod} import {node.name} as {node.asname}"

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
            stmt = self.format(node.target)
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
