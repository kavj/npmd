from collections import deque
from functools import singledispatchmethod
from typing import Optional, Set

import ir


def walk(node: ir.ValueRef):
    """
    yields all distinct sub-expressions and the base expression
    It was changed to include the base expression so that it's safe
    to walk without the need for explicit dispatch mechanics in higher level
    functions on Expression to disambiguate Expression vs non-Expression value
    references.

    :param node:
    :param seen:
    :return:
    """

    assert isinstance(node, ir.ValueRef)
    seen = set()
    enqueued = []
    enqueued.append(node)
    if isinstance(node, ir.Expression):
        # reverse subexpressions
        for subexpr in reversed(tuple(node.subexprs)):
            if subexpr not in seen:
                enqueued.append(subexpr)
    while enqueued:
        next_expr = enqueued.pop()
        if next_expr in seen:
            # this was already extended
            yield next_expr
        else:
            seen.add(next_expr)
            if isinstance(next_expr, ir.Expression):
                # if this is a not yet seen expression, first yield its unseen sub-expressions
                enqueued.append(next_expr)
                for subexpr in reversed(tuple(next_expr.subexprs)):
                    if subexpr not in seen:
                        enqueued.append(subexpr)
            else:
                # no subexpressions, so we can yield this now
                yield next_expr


def match_type(node: ir.ValueRef, types):
    """
    filtering version of walk
    :param node:
    :param types:
    :return:
    """
    for value in walk(node):
        if isinstance(value, types):
            yield value


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
