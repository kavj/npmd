import itertools
from collections import deque
from typing import Iterator, List, Union

import npmd.ir as ir

from npmd.utils import is_entry_point


def iterate_single_block(stmts: List[ir.StmtBase], start: int = 0):
    for stmt in itertools.islice(stmts, start, None):
        if is_entry_point(stmt):
            break
        yield stmt


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


def get_statement_lists(node: Union[ir.Function, ir.IfElse, ir.ForLoop, ir.WhileLoop, list],
                        enter_loops=True) -> Iterator[List[ir.StmtBase]]:
    """
    yields all statement lists by pre-ordering, breadth first
    :param node:
    :param enter_loops:
    :return:
    """

    queued = deque()
    if isinstance(node, ir.Function):
        queued.append(node.body)
    elif isinstance(node, ir.IfElse):
        queued.append(node.else_branch)
        queued.append(node.if_branch)
    elif isinstance(node, (ir.ForLoop, ir.WhileLoop)):
        if enter_loops:
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
                elif enter_loops:
                    queued.appendleft(stmt.body)
