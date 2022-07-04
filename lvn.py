import itertools

from collections import Counter, defaultdict
from functools import singledispatchmethod
from typing import List, Set, Union

import ir

from analysis import find_ephemeral_references
from symbol_table import SymbolTable
from type_checks import TypeHelper
from utils import is_entry_point, unpack_iterated
from traversal import depth_first_sequence_blocks, walk_parameters, BlockRewriter


def get_assign_counts(node: List[ir.StmtBase]):
    """
    This ignores in place assignments, should run after expansion
    :param node:
    :return:
    """
    counts = Counter()
    for stmt in itertools.chain(*depth_first_sequence_blocks(node)):
        if isinstance(stmt, ir.Assign):
            counts[stmt.target] += 1
        elif isinstance(stmt, ir.ForLoop):
            for target, _ in unpack_iterated(stmt.target, stmt.iterable):
                counts[target] += 1
    return counts


def get_uniquely_assigned(node: List[ir.StmtBase]):
    return {name for (name, count) in get_assign_counts(node).items() if count == 1}


def drop_unused_symbols(node: ir.Function, symbols: SymbolTable):
    """
    Purges any dead symbols from the symbol table. This is useful to remove non-ephemeral variables
    that are simply unused, so that they are not declared.
    :param node:
    :param symbols:
    :return:
    """
    counts = get_assign_counts(node.body)
    live_names = set(k.name for k in counts.keys() if isinstance(k, ir.NameRef))
    dead_names = set()
    for sym in symbols.symbols.values():
        if sym.is_local and not (sym.is_arg or sym.name in live_names):
            dead_names.add(sym.name)
    for name in dead_names:
        symbols.drop_symbol(name)


def get_last_assign_to_name(block: List[ir.StmtBase]):
    last_assign = {}
    # since this can be done via a generator, we need to reverse prior to this point
    for stmt in block:
        if isinstance(stmt, ir.Assign):
            if isinstance(stmt.target, ir.NameRef):
                if stmt.target not in last_assign:
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


class LiveStatementMarker:
    def __init__(self):
        self.live = set()
        self.latest = {}
        self.linked = defaultdict(set)

    def mark_name_live(self, name: ir.NameRef):
        last_assign = self.latest.get(name)
        if last_assign is None or last_assign in self.live:
            return
        queued = [last_assign]
        while queued:
            last_assign = queued.pop()
            if last_assign not in self.live:
                self.live.add(last_assign)
                queued.extend(self.linked[last_assign])

    def mark_live(self, expr: ir.ValueRef):
        for name in walk_parameters(expr):
            self.mark_name_live(name)

    def mark_potential_live_outs(self, ephemeral: Set[ir.NameRef]):
        for name, value in self.latest.items():
            if name not in ephemeral:
                self.mark_live(name)

    def mark_link(self, expr: ir.ValueRef, link_to: ir.Assign):
        for param in walk_parameters(expr):
            latest_param_assign = self.latest.get(param)
            if latest_param_assign is not None and latest_param_assign not in self.live:
                self.linked[link_to].add(latest_param_assign)

    @singledispatchmethod
    def mark(self, stmt: ir.StmtBase):
        assert isinstance(stmt, ir.StmtBase) and not is_entry_point(stmt)
        self.live.add(stmt)

    @mark.register
    def _(self, stmt: ir.Assign):
        target = stmt.target
        if isinstance(target, ir.Subscript):
            target_name = target.value
            latest_target_assign = self.latest.get(target_name)
            if latest_target_assign is None or target_name in self.live:
                self.mark_live(stmt.value)
                self.mark_live(stmt.target)
                self.live.add(stmt)
            else:
                self.mark_link(stmt.value, latest_target_assign)
                self.mark_link(stmt.target, latest_target_assign)
        else:
            self.latest[target] = stmt
            self.mark_link(stmt.value, stmt)

    @mark.register
    def _(self, stmt: ir.InPlaceOp):
        if isinstance(stmt.target, ir.NameRef):
            target_name = stmt.target
        else:
            target_name = stmt.target.value
        latest_assign = self.latest.get(target_name)
        if latest_assign is None or latest_assign in self.live:
            self.mark_live(stmt.value)
            self.live.add(stmt)
        else:
            # mark parameters as live if this statement is marked live
            self.mark_link(stmt.value, latest_assign)

    @mark.register
    def _(self, node: ir.SingleExpr):
        self.mark_live(node.value)

    @mark.register
    def _(self, node: ir.Return):
        if node.value is not None:
            self.mark_live(node.value)


def find_last_assign(block: List[ir.StmtBase]):
    last_assign = {}
    for stmt in reversed(block):
        if isinstance(stmt, ir.Assign) and isinstance(stmt.target, ir.NameRef):
            if stmt.target not in last_assign:
                last_assign[stmt.target] = stmt
    return last_assign


class AssignExpander(BlockRewriter):
    def __init__(self, symbols: SymbolTable):
        self.symbols = symbols

    def visit_block(self, block: List[ir.StmtBase]):
        """
        Expand in place assignments without requiring explicit type information.
        :param block:
        :return:
        """
        repl = []
        block = list(block)
        typer = TypeHelper(self.symbols)

        # we don't have to rely on determinations of what is ephemeral
        # if we know what augments a newly allocated handle
        clobbered = set()
        for stmt in block:
            if isinstance(stmt, ir.InPlaceOp):
                if isinstance(stmt.target, ir.NameRef):
                    if stmt.target in clobbered:
                        stmt = ir.Assign(stmt.target, stmt.value, stmt.pos)
                else:
                    if typer.is_scalar(stmt.target):
                        stmt = ir.Assign(stmt.target, stmt.value, stmt.pos)
            elif isinstance(stmt, ir.Assign):
                if isinstance(stmt.target, ir.NameRef):
                    clobbered.add(stmt.target)
            repl.append(stmt)
        return repl


class ExprRewriter:

    def __init__(self, symbols: SymbolTable):
        self.current = {}
        self.symbols = symbols

    def rewrite_expr(self, node: ir.ValueRef):
        if not isinstance(node, ir.ValueRef):
            msg = f'rewrite expression expected an expression, got "{node}"'
            raise TypeError(msg)
        assert isinstance(node, ir.ValueRef)
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


class Renamer(BlockRewriter):
    def __init__(self, symbols: SymbolTable, rename_targets: Set[ir.NameRef]):
        self.symbols = symbols
        self.rename_targets = rename_targets

    def visit_block(self, block: List[ir.StmtBase]):
        """
        Renames clobbering assignments. This is mainly intended for localizing ephemeral names.

        :param block:stmt = ir.Assign(stmt.target, stmt.value, stmt.pos)
        :return:
        """
        block = list(block)
        if not block:
            return block

        rewriter = ExprRewriter(self.symbols)
        repl = []

        for stmt in block:
            if isinstance(stmt, ir.Assign):
                value = rewriter.rewrite_expr(stmt.value)
                target = stmt.target
                if target in self.rename_targets:
                    target = rewriter.rename_target(target)
                else:
                    target = rewriter.rewrite_expr(target)
                stmt = ir.Assign(target, value, stmt.pos)
            elif isinstance(stmt, ir.InPlaceOp):
                value = rewriter.rewrite_expr(stmt.value)
                target = rewriter.rewrite_expr(stmt.target)
                stmt = ir.InPlaceOp(target, value, stmt.pos)
            elif isinstance(stmt, ir.Return):
                if stmt.value is not None:
                    value = rewriter.rewrite_expr(stmt.value)
                    stmt = ir.Return(value, stmt.pos)
            elif isinstance(stmt, ir.SingleExpr):
                value = rewriter.rewrite_expr(stmt.value)
                stmt = ir.SingleExpr(value, stmt.pos)
            repl.append(stmt)
        return repl


class DeadAssignElim(BlockRewriter):

    def __init__(self,  ephemeral: Set[ir.NameRef]):
        self.ephemeral = ephemeral

    def visit_block(self, block: List[ir.StmtBase]):
        """
        Checks for assignments to control block scoped variables that are not used in any statement other than
        a self reference.
        :param block:
        :return:
        """

        if is_entry_point(block[0]):
            return block

        marker = LiveStatementMarker()
        for stmt in block:
            marker.mark(stmt)

        # mark anything that could escape as live
        marker.mark_potential_live_outs(self.ephemeral)

        # Now rewrite the block
        repl = []

        for stmt in block:
            if isinstance(stmt, (ir.Assign, ir.InPlaceOp)):
                if stmt in marker.live:
                    repl.append(stmt)
            else:
                repl.append(stmt)

        return repl


def remove_dead(node: ir.Function, symbols: SymbolTable):
    # now remove any dead assignments
    ephemeral = find_ephemeral_references(node.body, symbols)
    ephemeral.difference_update(node.args)
    # remove any unreferenced assignments
    dce = DeadAssignElim(ephemeral)
    body = dce.visit(node.body)
    func = ir.Function(node.name, node.args, body)
    drop_unused_symbols(func, symbols)
    return func


def optimize_statements(node: ir.Function, symbols: SymbolTable):
    """
    Function scope dead code removal pass
    :param node:
    :param symbols:
    :return:
    """
    # first expand in place assignments where possible
    assign_expander = AssignExpander(symbols)
    body = assign_expander.visit(node.body)
    # now remove any dead assignments
    ephemeral = find_ephemeral_references(body, symbols)
    ephemeral.difference_update(node.args)
    # remove any unreferenced assignments
    dce = DeadAssignElim(ephemeral)
    body = dce.visit(body)
    #
    rename_targets = find_ephemeral_references(body, symbols)
    rename_targets.difference_update(get_uniquely_assigned(body))
    rename_targets.difference_update(node.args)
    block_renamer = Renamer(symbols, rename_targets)
    body = block_renamer.visit(body)

    return ir.Function(node.name, node.args, body)
