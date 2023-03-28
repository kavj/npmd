from typing import Set, Tuple

import lib.ir as ir

from collections import Counter
from functools import singledispatch

from lib.blocks import BasicBlock, FunctionContext
from lib.expression_walkers import walk_parameters
from lib.graph_walkers import get_reduced_graph, get_reachable_nodes
from lib.unpacking import unpack_loop_iter


def get_assign_counts(func: FunctionContext, entry_point: BasicBlock):
    """
    Count the number of times each variable is assigned within the function body
    :param func:
    :param entry_point:
    :return:
    """
    assert entry_point.is_entry_point
    assign_counts = Counter()
    graph = get_reduced_graph(func)
    for stmt in get_reachable_nodes(graph, entry_point):
        if isinstance(stmt, ir.Assign) and isinstance(stmt.target, ir.NameRef):
            assign_counts[stmt.target] += 1
        elif isinstance(stmt, ir.ForLoop):
            for target, _ in unpack_loop_iter(stmt):
                # it's okay to count duplicates in unpacking
                assign_counts[target] += 1
    return assign_counts


def get_assigned_or_augmented(func: FunctionContext, node: BasicBlock) -> Tuple[Set[ir.NameRef], Set[ir.NameRef]]:
    bound = set()
    augmented = set()
    for stmt in get_reachable_nodes(func.graph, node):
        if isinstance(stmt, ir.Assign):
            if isinstance(stmt.target, ir.NameRef):
                bound.add(stmt.target)
            elif isinstance(stmt.target, ir.Subscript):
                augmented.add(stmt.target.value)
        elif isinstance(stmt, ir.InPlaceOp):
            if isinstance(stmt.target, ir.NameRef):
                augmented.add(stmt.target)
            else:
                assert isinstance(stmt.target, ir.Subscript)
                if isinstance(stmt.target.value, ir.NameRef):
                    augmented.add(stmt.target.value)
        elif isinstance(stmt, ir.ForLoop):
            # nested loop should not clobber
            for target, _ in unpack_loop_iter(stmt):
                if isinstance(target, ir.NameRef):
                    bound.add(target)
    return bound, augmented


@singledispatch
def get_assigned(node):
    raise TypeError()


@get_assigned.register
def _(node: ir.StmtBase):
    return  # avoid yielding None
    yield


@get_assigned.register
def _(node: ir.Assign):
    if isinstance(node.target, ir.NameRef):
        yield node.target


@get_assigned.register
def _(node: ir.ForLoop):
    for target, value in unpack_loop_iter(node):
        if isinstance(target, ir.NameRef):
            yield target


@get_assigned.register
def _(node: ir.Function):
    return
    yield


@singledispatch
def get_referenced(node):
    msg = f'No method to extract referenced for {node}.'
    raise TypeError(msg)


@get_referenced.register
def _(node: ir.Function):
    return
    yield


@get_referenced.register
def _(node: ir.StmtBase):
    return
    yield


@get_referenced.register
def _(node: ir.Assign):
    yield from walk_parameters(node.value)
    if not isinstance(node.target, ir.NameRef):
        yield from walk_parameters(node.target)


@get_referenced.register
def _(node: ir.InPlaceOp):
    yield from walk_parameters(node.value)


@get_referenced.register
def _(node: ir.IfElse):
    yield from walk_parameters(node.test)


@get_referenced.register
def _(node: ir.WhileLoop):
    yield from walk_parameters(node.test)


@get_referenced.register
def _(node: ir.SingleExpr):
    yield from walk_parameters(node.value)


@get_referenced.register
def _(node: ir.Return):
    yield from walk_parameters(node.value)


@get_referenced.register
def _(node: ir.ForLoop):
    for target, value in unpack_loop_iter(node):
        yield from walk_parameters(value)
        if not isinstance(target, ir.NameRef):
            yield from walk_parameters(target)


@singledispatch
def get_expressions(node):
    msg = f'No method to extract expressions from {node}'
    raise TypeError(msg)


@get_expressions.register
def _(node: ir.StmtBase):
    return
    yield


@get_expressions.register
def _(node: ir.Assign):
    yield node.value
    if not isinstance(node.target, ir.NameRef):
        yield node.target


@get_expressions.register
def _(node: ir.ForLoop):
    for target, iterable in unpack_loop_iter(node):
        yield iterable
        yield target


@get_expressions.register
def _(node: ir.IfElse):
    yield node.test


@get_expressions.register
def _(node: ir.InPlaceOp):
    yield node.value


@get_expressions.register
def _(node: ir.SingleExpr):
    yield node.value


@get_expressions.register
def _(node: ir.Return):
    yield node.value


@get_expressions.register
def _(node: ir.WhileLoop):
    yield node.test
