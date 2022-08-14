import itertools

from functools import singledispatchmethod
from typing import Dict, Iterable, Optional, Union

import npmd.ir as ir

from npmd.blocks import sequence_block_intervals
from npmd.liveness import BlockLiveness
from npmd.symbol_table import SymbolTable
from npmd.traversal import walk_nodes
from npmd.type_checks import TypeHelper
from npmd.utils import is_entry_point


def get_last_assign_to_name(block: Iterable[ir.StmtBase]):
    last_assign = {}
    # since this can be done via a generator, we need to reverse prior to this point
    for stmt in block:
        if isinstance(stmt, ir.Assign):
            if isinstance(stmt.target, ir.NameRef):
                last_assign[stmt.target] = stmt
    return last_assign


def get_target_name(stmt: Union[ir.Assign, ir.InPlaceOp]):
    target = stmt.target
    if isinstance(target, ir.NameRef):
        return target
    elif isinstance(target, ir.Subscript):
        return target.value
    else:
        msg = f'Cannot extract target name from "{target}".'
        raise TypeError(msg)


class ExprRewriter:

    def __init__(self, symbols: SymbolTable, current: Optional[Dict[ir.NameRef, ir.NameRef]] = None):
        self.current = {} if current is None else current
        self.symbols = symbols

    def rewrite_expr(self, node: ir.ValueRef):
        if not isinstance(node, ir.ValueRef):
            msg = f'rewrite expression expected an expression, got "{node}"'
            raise TypeError(msg)
        if isinstance(node, ir.Expression):
            repl = node.reconstruct(*(self.rewrite_expr(subexpr) for subexpr in node.subexprs))
            if repl != node:
                return repl
            else:  # don't propagate identical copies
                return node
        return self.current.get(node, node)

    def rename_target(self, target: ir.NameRef):
        assert isinstance(target, ir.NameRef)
        t = self.symbols.check_type(target, allow_none=True)
        renamed = self.symbols.make_unique_name_like(target, t)
        self.current[target] = renamed
        return renamed

    @singledispatchmethod
    def rewrite(self, node):
        raise TypeError(str(node))

    @rewrite.register
    def _(self, node: ir.StmtBase):
        return node

    @rewrite.register
    def _(self, node: ir.Assign):
        value = self.rewrite_expr(node.value)
        if isinstance(node.target, ir.NameRef):
            target = self.rename_target(node.target)
        else:
            target = self.rewrite_expr(node.target)
        repl = ir.Assign(target, value, node.pos)
        if repl != node:
            return repl
        return node

    @rewrite.register
    def _(self, node: ir.InPlaceOp):
        value = self.rewrite_expr(node.value)
        if isinstance(node.target, ir.NameRef):
            # rename anyway
            target = self.rename_target(node.target)
            return ir.Assign(target, value, node.pos)
        else:
            target = self.rewrite_expr(node.target)
            return ir.InPlaceOp(target, value, node.pos)

    @rewrite.register
    def _(self, node: ir.SingleExpr):
        return ir.SingleExpr(self.rewrite_expr(node.value), node.pos)

    @rewrite.register
    def _(self, node: ir.Return):
        return ir.Return(self.rewrite_expr(node.value), node.pos)


class ExprInliner:

    def __init__(self, symbols: SymbolTable):
        self.current = {}
        self.symbols = symbols
        self.typer = TypeHelper(symbols)

    def rewrite_expr(self, node: ir.ValueRef):
        if not isinstance(node, ir.ValueRef):
            msg = f'rewrite expression expected an expression, got "{node}"'
            raise TypeError(msg)
        if isinstance(node, ir.Expression):
            repl = node.reconstruct(*(self.rewrite_expr(subexpr) for subexpr in node.subexprs))
            if repl != node:
                return repl
            else:  # don't propagate identical copies
                return node
        elif isinstance(node, ir.NameRef):
            return self.current.get(node, node)
        else:
            # should be a constant
            return node

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
        value = self.rewrite_expr(node.value)
        if isinstance(node.target, ir.NameRef):
            # adds casting if necessary
            value = self.bind_target(node.target, node.value)
        else:
            target = self.rewrite_expr(node.target)
        repl = ir.Assign(target, value, node.pos)
        if repl != node:
            return repl
        # if no changes persist, return the original rather than a copy
        return node

    @rewrite.register
    def _(self, node: ir.InPlaceOp):
        target = node.target
        value = self.rewrite_expr(node.value)
        if isinstance(node.target, ir.NameRef):
            # rename anyway
            value = self.bind_target(target, value)
        repl = ir.InPlaceOp(target, value, node.pos)
        if repl != node:
            return repl
        return node

    @rewrite.register
    def _(self, node: ir.SingleExpr):
        value = self.rewrite_expr(node.value)
        repl = ir.SingleExpr(value, node.pos)
        if repl != node:
            return repl
        return repl

    @rewrite.register
    def _(self, node: ir.Return):
        return ir.Return(self.rewrite_expr(node.value), node.pos)

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


def get_leading_block_statements(stmts: Iterable[ir.Statement]):
    start, stop = next(sequence_block_intervals(stmts))
    leading = stmts[start:stop]
    return leading


def find_branch_nested_loops(node: ir.IfElse):
    nested = []
    for stmt in itertools.chain(walk_nodes(node.if_branch), walk_nodes(node.else_branch)):
        if isinstance(stmt, (ir.ForLoop, ir.WhileLoop)):
            nested.append(stmt)
    return nested


def find_uniform_exprs():
    pass


def branch_inlining(func: ir.Function, header: ir.IfElse, symbols: SymbolTable):
    for branch in (header.if_branch, header.else_branch):
        repl = []
        rewriter = ExprInliner(symbols)

        for stmt in branch:
            if is_entry_point(stmt):
                assert isinstance(stmt, ir.IfElse)
                flattened = branch_inlining(func, header, symbols)
                for substmt in flattened:
                    substmt = rewriter.rewrite(substmt)
                    repl.append(substmt)

        branch.clear()
        branch.extend(repl)

    return ()


def branch_localize(if_branch: Iterable[ir.Statement], else_branch: Iterable[ir.Statement], symbols: SymbolTable, if_live: BlockLiveness, else_live: BlockLiveness):
    # worth noting, we need liveness information here, since we can avoid blending anything that is dead no exit
    if_repl = []
    else_repl = []
    if_lead = get_leading_block_statements(if_branch)
    else_lead = get_leading_block_statements(else_branch)
    last = get_last_assign_to_name(if_lead)
    rewriter = ExprRewriter(symbols)
    for stmt in if_lead:
        if isinstance(stmt, (ir.Assign, ir.InPlaceOp)):
            if isinstance(stmt.target, ir.NameRef):
                if stmt.target not in if_live.live_out or last[stmt.target] != stmt:
                    # we should be checking for already created identical values, since these won't be live out
                    rewriter.rewrite()
        if_repl.append(rewriter.rewrite(stmt))

    for stmt in else_lead:
        else_repl.append(rewriter.rewrite(stmt))
