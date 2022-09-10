from functools import singledispatchmethod
from typing import Dict

import npmd.ir as ir

from npmd.analysis import get_assign_counts
from npmd.blocks import build_function_graph
from npmd.liveness import find_ephemeral_assigns, find_live_in_out
from npmd.symbol_table import SymbolTable
from npmd.type_checks import TypeHelper


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


class ExprInliner:

    def __init__(self, symbols: SymbolTable):
        self.current = {}
        self.symbols = symbols
        self.typer = TypeHelper(symbols)

    def bind_target(self, target: ir.NameRef, value: ir.ValueRef):
        assert isinstance(target, ir.NameRef)
        target_type = self.symbols.check_type(target, allow_none=True)
        value_type = self.typer(value)
        if target_type != value_type:
            # Todo: need can cast validation for these things..
            value = ir.CAST(value, target_type)
        self.current[target] = value
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
        value = rewrite_expr(self.current, node.value)
        if isinstance(node.target, ir.NameRef):
            # adds casting if necessary
            value = self.bind_target(node.target, node.value)
        else:
            target = rewrite_expr(self.current, node.target)
        repl = ir.Assign(target, value, node.pos)
        if repl != node:
            return repl
        # if no changes persist, return the original rather than a copy
        return node

    @rewrite.register
    def _(self, node: ir.InPlaceOp):
        target = node.target
        value = rewrite_expr(self.current, node.value)
        if isinstance(node.target, ir.NameRef):
            # rename anyway
            value = self.bind_target(target, value)
        repl = ir.InPlaceOp(target, value, node.pos)
        if repl != node:
            return repl
        return node

    @rewrite.register
    def _(self, node: ir.SingleExpr):
        value = rewrite_expr(self.current, node.value)
        repl = ir.SingleExpr(value, node.pos)
        if repl != node:
            return repl
        return repl

    @rewrite.register
    def _(self, node: ir.Return):
        return ir.Return(rewrite_expr(self.current, node.value), node.pos)

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
