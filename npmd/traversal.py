from collections import deque
from functools import singledispatch
from typing import Iterable, Iterator, List, Union

import npmd.ir as ir

from npmd.utils import is_entry_point, unpack_iterated


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


def get_statement_lists(node: Union[ir.Function, ir.IfElse, ir.ForLoop, ir.WhileLoop, list]) -> Iterator[List[ir.StmtBase]]:
    """
    yields all statement lists by pre-ordering, breadth first
    :param node:
    :return:
    """

    queued = deque()
    if isinstance(node, ir.Function):
        queued.append(node.body)
    elif isinstance(node, ir.IfElse):
        queued.append(node.else_branch)
        queued.append(node.if_branch)
    elif isinstance(node, (ir.ForLoop, ir.WhileLoop)):
        queued.append(node.body)
    else:  # statement list
        queued.append(node)
    while queued:
        stmts = queued.pop()
        # yield in case caller modifies trim
        yield stmts
        for stmt in stmts:
            if is_entry_point(stmt):
                if isinstance(stmt, ir.IfElse):
                    queued.appendleft(stmt.if_branch)
                    queued.appendleft(stmt.else_branch)
                else:
                    queued.appendleft(stmt.body)


def all_entry_points(stmts: Iterable[ir.StmtBase]):
    for stmt in walk_nodes(stmts):
        if isinstance(stmt, (ir.IfElse, ir.ForLoop, ir.WhileLoop)):
            yield stmt


def all_loops(stmts: Iterable[ir.StmtBase]):
    for stmt in walk_nodes(stmts):
        if isinstance(stmt, (ir.ForLoop, ir.WhileLoop)):
            yield stmt


@singledispatch
def extract_expressions(node):
    msg = f'No method to extract expressions from {node}'
    raise TypeError(msg)


@extract_expressions.register
def _(node: ir.StmtBase):
    yield


@extract_expressions.register
def _(node: ir.Assign):
    yield node.value
    yield node.target


@extract_expressions.register
def _(node: ir.Case):
    for test in node.conditions:
        yield test


@extract_expressions.register
def _(node: ir.ForLoop):
    for target, iterable in unpack_iterated(node.target, node.iterable):
        yield iterable
        yield target


@extract_expressions.register
def _(node: ir.IfElse):
    yield node.test


@extract_expressions.register
def _(node: ir.InPlaceOp):
    yield node.value


@extract_expressions.register
def _(node: ir.SingleExpr):
    yield node.value


@extract_expressions.register
def _(node: ir.Return):
    yield node.value


@extract_expressions.register
def _(node: ir.WhileLoop):
    yield node.test
