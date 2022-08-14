from functools import singledispatchmethod
from typing import Iterable, Union

import npmd.ir as ir

from npmd.utils import unpack_iterated


def get_nested(node: Union[ir.IfElse, ir.ForLoop, ir.WhileLoop]):
    if isinstance(node, ir.IfElse):
        yield node.if_branch
        yield node.else_branch
    elif isinstance(node, (ir.ForLoop, ir.WhileLoop)):
        yield node.body


def walk(node: ir.ValueRef):
    """
    yields all distinct sub-expressions and the base expression
    It was changed to include the base expression so that it's safe
    to walk without the need for explicit dispatch mechanics in higher level
    functions on Expression to disambiguate Expression vs non-Expression value
    references.

    :param node:
    :return:
    """
    if not isinstance(node, ir.ValueRef):
        msg = f'walk expects a value ref. Received: "{node}"'
        raise TypeError(msg)
    assert isinstance(node, ir.ValueRef)
    if isinstance(node, ir.Expression):
        seen = {node}
        enqueued = [(node, node.subexprs)]
        while enqueued:
            expr, subexprs = enqueued[-1]
            for subexpr in subexprs:
                if subexpr in seen:
                    continue
                seen.add(subexpr)
                if isinstance(subexpr, ir.Expression):
                    enqueued.append((subexpr, subexpr.subexprs))
                    break
                yield subexpr
            else:
                # exhausted
                yield expr
                enqueued.pop()
    else:
        yield node


def walk_parameters(node: ir.ValueRef):
    for value in walk(node):
        if isinstance(value, ir.NameRef):
            yield value


def walk_nodes(stmts: Iterable[ir.StmtBase]):
    queued = [iter(stmts)]
    seen = set()
    while queued:
        for stmt in queued[-1]:
            stmt_id = id(stmt)
            if stmt_id in seen:
                # If we are visiting out of order, this avoids
                # visiting anything from an outer scope that
                # was seen natively
                continue
            seen.add(stmt_id)
            yield stmt
            if isinstance(stmt, ir.IfElse):
                queued.append(iter(stmt.if_branch))
                queued.append(iter(stmt.else_branch))
                break
            elif isinstance(stmt, (ir.ForLoop, ir.WhileLoop)):
                queued.append(iter(stmt.body))
                break
        else:
            queued.pop()


def all_entry_points(stmts: Iterable[ir.StmtBase]):
    for stmt in walk_nodes(stmts):
        if isinstance(stmt, (ir.IfElse, ir.ForLoop, ir.WhileLoop)):
            yield stmt


def all_loops(stmts: Iterable[ir.StmtBase]):
    for stmt in walk_nodes(stmts):
        if isinstance(stmt, (ir.ForLoop, ir.WhileLoop)):
            yield stmt


class ExtractExprs:

    def __call__(self, node):
        return self.visit(node)

    @singledispatchmethod
    def visit(self, node):
        msg = f'No method to extract expressions from {node}'
        raise TypeError(msg)

    @visit.register
    def _(self, node: ir.StmtBase):
        yield

    @visit.register
    def _(self, node: ir.Assign):
        if isinstance(node.value, ir.Expression):
            yield node.value
        if isinstance(node.target, ir.Expression):
            yield node.target

    @visit.register
    def _(self, node: ir.InPlaceOp):
        if isinstance(node.value, ir.Expression):
            yield node.value

    @visit.register
    def _(self, node: ir.SingleExpr):
        if isinstance(node.value, ir.Expression):
            yield node.value

    @visit.register
    def _(self, node: ir.IfElse):
        if isinstance(node.test, ir.Expression):
            yield node.test

    @visit.register
    def _(self, node: ir.ForLoop):
        for _, iterable in unpack_iterated(node.target, node.iterable):
            if isinstance(iterable, ir.Expression):
                yield iterable

    @visit.register
    def _(self, node: ir.WhileLoop):
        if isinstance(node.test, ir.Expression):
            yield node.test

    @visit.register
    def _(self, node: ir.Case):
        for test in node.conditions:
            if isinstance(test, ir.Expression):
                yield test
