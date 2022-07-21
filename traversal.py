from typing import Iterable, Tuple, Union

import ir


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
    while queued:
        for stmt in queued[-1]:
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
