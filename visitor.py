import typing
from functools import singledispatchmethod

import ir


def walk(node):
    """
    walk an expression depth first in post order, yielding everything but the original node
    """
    if hasattr(node, "subexprs"):
        for subexpr in node.subexprs:
            yield from walk(subexpr)
            yield subexpr


def walk_unique(node):
    seen = set()
    for subexpr in walk(node):
        if subexpr not in seen:
            yield subexpr
            seen.add(subexpr)


class ExpressionVisitor:
    """
    Base class for an expression visitor for cases that do not reconstruct their arguments.

    """

    def __call__(self, expr):
        return self.visit(expr)

    def visit(self, expr):
        if isinstance(expr, ir.Expression):
            for subexpr in expr.subexprs:
                self.visit(subexpr)
        elif not isinstance(expr, ir.ValueRef):
            msg = f"No method to rewrite object of type {type(expr)}."
            raise NotImplementedError(msg)


class ExpressionTransformer:
    """
    Base class for an expression visitor that reconstructs an argument expression from sub-expressions.

    """

    def __call__(self, node):
        return self.visit(node)

    @singledispatchmethod
    def visit(self, node):
        msg = f"No method to visit node of type {type(node)}"
        raise TypeError(msg)

    @visit.register
    def _(self, node: ir.ValueRef):
        return node

    # array initializers use these
    @visit.register
    def _(self, node: ir.ScalarType):
        return node

    @visit.register
    def _(self, node: ir.Expression):
        cls = type(node)
        return cls(*(self.visit(subexpr) for subexpr in node.subexprs))


class StmtVisitor:

    @singledispatchmethod
    def visit(self, node):
        msg = f"No handler exists for node type {type(node)}."
        raise NotImplementedError(msg)

    @visit.register
    def _(self, node: list):
        for stmt in node:
            self.visit(stmt)
            # ignore unreachable
            if isinstance(stmt, (ir.Continue, ir.Break, ir.Return)):
                break

    @visit.register
    def _(self, node: ir.Function):
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.StmtBase):
        pass

    @visit.register
    def _(self, node: ir.IfElse):
        self.visit(node.if_branch)
        self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.ForLoop):
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.WhileLoop):
        self.visit(node.body)


class StmtTransformer:

    @singledispatchmethod
    def visit(self, node):
        msg = f"No handler for node type {type(node)}."
        raise NotImplementedError(msg)

    @visit.register
    def _(self, node: list):
        repl = []
        for stmt in node:
            stmt = self.visit(stmt)
            repl.append(stmt)
        if repl != node:
            # Only return a copy if it differs from the input.
            node = repl
        return node

    @visit.register
    def _(self, node: ir.Function):
        body = self.visit(node.body)
        if body != node.body:
            return ir.Function(node.name, node.args, body)
        return node

    @visit.register
    def _(self, node: ir.StmtBase):
        return node

    @visit.register
    def _(self, node: ir.IfElse):
        if_branch = self.visit(node.if_branch)
        else_branch = self.visit(node.else_branch)
        if if_branch is not node.if_branch or else_branch is not node.else_branch:
            node = ir.IfElse(node.test, if_branch, else_branch, node.pos)
        return node

    @visit.register
    def _(self, node: ir.ForLoop):
        body = self.visit(node.body)
        if body != node.body:
            node = ir.ForLoop(node.target, node.iterable, body, node.pos)
        return node

    @visit.register
    def _(self, node: ir.WhileLoop):
        body = self.visit(node.body)
        if body is not node.body:
            node = ir.WhileLoop(node.test, body, node.pos)
        return node
