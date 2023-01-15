import networkx as nx

import lib.ir as ir


def walk_graph(graph: nx.DiGraph, entry_point, reverse=False):
    if reverse:
        graph = nx.reverse_view(graph)
    return nx.dfs_preorder_nodes(graph, entry_point)


def walk_expr(node: ir.ValueRef):
    """
    yields all distinct sub-expressions and the base expression based on post order
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
    for value in walk_expr(node):
        if isinstance(value, ir.NameRef):
            yield value
