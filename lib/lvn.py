from functools import singledispatch, singledispatchmethod
from typing import Dict, Iterable, Union

import lib.ir as ir

from lib.analysis import get_assign_counts
from lib.blocks import build_function_graph
from lib.liveness import find_ephemeral_assigns, find_live_in_out
from lib.symbol_table import SymbolTable
from lib.type_checks import TypeHelper
from lib.utils import is_basic_assign


def rewrite_expr(current: Dict[ir.ValueRef, ir.ValueRef], node: ir.ValueRef):
    if not isinstance(node, ir.ValueRef):
        msg = f'rewrite expression expected an expression, got "{node}"'
        raise TypeError(msg)
    if isinstance(node, ir.Expression):
        # Todo: this path needs real testing
        repl = node.reconstruct(*(rewrite_expr(current, subexpr) for subexpr in node.subexprs))
        if repl != node:
            return repl
        else:  # don't propagate identical copies
            return node
    elif isinstance(node, ir.NameRef):
        return current.get(node, node)
    else:
        return node


# statement rewriter doesn't create new names here

def rename_ephemeral(func: ir.Function, symbols: SymbolTable):
    # Helpful for weakening the need for type inference in cases where a type is not declared.
    graph = build_function_graph(func)
    liveness = find_live_in_out(graph)
    ephemeral = find_ephemeral_assigns(liveness)
    uniquely_assigned = {name for (name, count) in get_assign_counts(func).items() if count == 1}
    # avoid versioning uniquely assigned names
    ephemeral = {stmt for stmt in ephemeral if stmt.target not in uniquely_assigned}
    latest = {}
    for block in graph.nodes():
        if block.is_entry_point:
            continue
        for index, stmt in enumerate(block.statements, block.start):
            pos = stmt.pos
            if isinstance(stmt, ir.Assign):
                if not latest and stmt not in ephemeral:
                    continue
                value = rewrite_expr(latest, stmt.value)
                if stmt in ephemeral:
                    target = symbols.make_versioned(stmt.target)
                    latest[stmt.target] = target
                else:
                    target = rewrite_expr(latest, stmt.target)
                block.statements[index] = ir.Assign(target, value, pos)
            elif not latest:
                continue
            else:
                if isinstance(stmt, ir.InPlaceOp):
                    value = rewrite_expr(latest, stmt.value)
                    target = rewrite_expr(latest, stmt.target)
                    block.statements[index] = ir.InPlaceOp(target, value, stmt.pos)
                elif isinstance(stmt, ir.SingleExpr):
                    value = rewrite_expr(latest, stmt.value)
                    block.statements[index] = ir.SingleExpr(value, stmt.pos)
                elif isinstance(stmt, ir.Return):
                    value = rewrite_expr(latest, stmt.value)
                    block.statements[index] = ir.Return(value, stmt.pos)
        latest.clear()


def get_last_basic_assign(stmts: Iterable[Union[ir.StmtBase, ir.Assign]]):
    # simple because it doesn't rely on reversing partial views, eg itertools.islice
    last = {}
    for stmt in stmts:
        if is_basic_assign(stmt):
            last[stmt.target] = stmt
    return last


class ExprRenamer:

    def __init__(self, symbols: SymbolTable, last_assign: Dict[ir.NameRef, ir.Assign]):
        self.latest = {}
        self.last_assign = last_assign
        self.symbols = symbols
        self.typer = TypeHelper(symbols)

    def bind_target(self, target: ir.NameRef, value: ir.ValueRef):
        assert isinstance(target, ir.NameRef)
        target_type = self.symbols.check_type(target, allow_none=True)
        value_type = self.typer(value)
        if target_type != value_type:
            value = ir.CAST(value, target_type)
        self.latest[target] = value
        return value

    @singledispatchmethod
    def rewrite(self, node):
        raise TypeError(str(node))

    @rewrite.register
    def _(self, node: ir.StmtBase):
        return node

    @rewrite.register
    def _(self, node: ir.Assign):
        target = node.target
        # inline prior expressions
        value = rewrite_expr(self.latest, node.value)
        if is_basic_assign(node):
            if node is self.last_assign[node.target]:
                # ensure
                self.latest[node.target] = node.target
            else:
                value = self.bind_target(node.target, node.value)
        else:
            target = rewrite_expr(self.latest, node.target)
        if target == node.target and value == node.value:
            return node
        return ir.Assign(target, value, node.pos)

    @rewrite.register
    def _(self, node: ir.InPlaceOp):
        target = node.target
        value = rewrite_expr(self.latest, node.value)
        if isinstance(node.target, ir.NameRef):
            # rename anyway
            value = self.bind_target(target, value)
        if target == node.target and value == node.value:
            return node
        return ir.InPlaceOp(target, value, node.pos)

    @rewrite.register
    def _(self, node: ir.SingleExpr):
        value = rewrite_expr(self.latest, node.value)
        if value == node.value:
            return node
        return ir.SingleExpr(value, node.pos)

    @rewrite.register
    def _(self, node: ir.Return):
        return ir.Return(rewrite_expr(self.latest, node.value), node.pos)

    @rewrite.register
    def _(self, node: ir.IfElse):
        msg = f'Cannot be applied when crossing a branch bound, position {node.pos}.'
        raise TypeError(msg)

    @rewrite.register
    def _(self, node: ir.ForLoop):
        msg = f'Cannot inline across possible loop boundary, position: {node.pos}. This is probably a bug somewhere.'
        raise TypeError(msg)

    @rewrite.register
    def _(self, node: ir.WhileLoop):
        msg = f'Cannot inline across possible loop boundary, position: {node.pos}. This is probably a bug somewhere.'
        raise TypeError(msg)


@singledispatch
def rewrite_statement(node, latest: Dict[ir.NameRef, ir.NameRef]):
    raise TypeError


@rewrite_statement.register
def _(node: ir.StmtBase, latest: Dict[ir.NameRef, ir.NameRef]):
    return node


@rewrite_statement.register
def _(node: ir.Assign, latest: Dict[ir.NameRef, ir.NameRef]):
    target = node.target
    # inline prior expressions
    value = rewrite_expr(latest, node.value)
    if not is_basic_assign(node):
        target = rewrite_expr(latest, node.target)
    if target == node.target and value == node.value:
        return node
    return ir.Assign(target, value, node.pos)


@rewrite_statement.register
def _(node: ir.InPlaceOp, latest: Dict[ir.NameRef, ir.NameRef]):
    target = rewrite_expr(latest, node.target)
    value = rewrite_expr(latest, node.value)
    if target == node.target and value == node.value:
        return node
    return ir.InPlaceOp(target, value, node.pos)


@rewrite_statement.register
def _(self, node: ir.SingleExpr):
    value = rewrite_expr(self.latest, node.value)
    if value == node.value:
        return node
    return ir.SingleExpr(value, node.pos)


@rewrite_statement.register
def _(node: ir.Return, latest: Dict[ir.NameRef, ir.NameRef]):
    value = rewrite_expr(latest, node.value)
    if value == node.value:
        return node
    return ir.Return(value, node.pos)


@rewrite_statement.register
def _(node: ir.IfElse):
    msg = f'Cannot be applied when crossing a branch bound, position {node.pos}.'
    raise TypeError(msg)


@rewrite_statement.register
def _(node: ir.ForLoop):
    msg = f'Cannot inline across possible loop boundary, position: {node.pos}. This is probably a bug somewhere.'
    raise TypeError(msg)


@rewrite_statement.register
def _(node: ir.WhileLoop):
    msg = f'Cannot inline across possible loop boundary, position: {node.pos}. This is probably a bug somewhere.'
    raise TypeError(msg)


def get_target_name(node: ir.ValueRef):
    if isinstance(node, ir.NameRef):
        return node
    elif isinstance(node, ir.Subscript):
        while isinstance(node, ir.Subscript):
            node = node.value
        if isinstance(node, ir.NameRef):
            return node
