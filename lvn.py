import itertools
from collections import Counter, defaultdict
from functools import partial, singledispatch, singledispatchmethod
from typing import Callable, Dict, Iterable, List, Set, Union

import ir

from analysis import find_ephemeral_references
from symbol_table import SymbolTable
from type_checks import TypeHelper
from utils import is_entry_point, unpack_iterated
from traversal import depth_first_sequence_statements, sequence_block_intervals, walk_parameters, StmtVisitor, \
     BlockRewriter


def get_assign_counts(node: Iterable[ir.StmtBase]):
    """
    This ignores in place assignments, should run after expansion
    :param node:
    :return:
    """
    counts = Counter()
    for stmt in depth_first_sequence_statements(node):
        if isinstance(stmt, ir.Assign):
            counts[stmt.target] += 1
        elif isinstance(stmt, ir.ForLoop):
            for target, _ in unpack_iterated(stmt.target, stmt.iterable):
                counts[target] += 1
    return counts


def get_block_assign_counts(block: Iterable[ir.StmtBase]):
    counts = Counter()
    for stmt in block:
        if isinstance(stmt, ir.Assign):
            counts[stmt.target] += 1
    return counts


def get_uniquely_assigned(node: Iterable[ir.StmtBase]):
    return {name for (name, count) in get_assign_counts(node).items() if count == 1}


def strip_dead_symbols(node: ir.Function, symbols: SymbolTable):
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
    for sym in symbols.all_locals:
        if not (sym.is_arg or sym.name in live_names):
            dead_names.add(sym.name)
    for name in dead_names:
        symbols.drop_symbol(name)


def get_last_assign_to_name(block: Iterable[ir.StmtBase]):
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


def dead_assign_elim(ephemeral: Set[ir.NameRef], block: List[ir.StmtBase]):
    """
    Checks for assignments to control block scoped variables that are not used in any statement other than
    a self reference.
    :param ephemeral:
    :param block:
    :return:
    """
    if not isinstance(block, list):
        block = list(block)

    if is_entry_point(block[0]):
        return block

    marker = LiveStatementMarker()
    for stmt in block:
        marker.mark(stmt)

    # mark anything that could escape as live
    marker.mark_potential_live_outs(ephemeral)

    # Now rewrite the block
    repl = []

    for stmt in block:
        if isinstance(stmt, (ir.Assign, ir.InPlaceOp)):
            if stmt in marker.live:
                repl.append(stmt)
        else:
            repl.append(stmt)

    return repl


def run_func_local_opts(node: ir.Function, symbols: SymbolTable):
    # do this before checking ephemeral references, since it may reveal more
    node = ExpandAssigns(symbols).visit(node)
    node = NameLocalizer.localize_names(node, symbols)
    ephemeral_refs = find_ephemeral_references(node.body, symbols)
    ephemeral_refs.difference_update(node.args)
    # have to handle sequencing manually here for context
    body = run_local_opts(node.body, symbols, ephemeral_refs)
    return ir.Function(node.name, node.args, body)


def expand_assigns(block: Iterable[ir.StmtBase]):
    """
    Expand in place assignments without requiring explicit type information.
    :param block:
    :return:
    """
    repl = []
    block = list(block)

    # we don't have to rely on determinations of what is ephemeral
    # if we know what augments a newly allocated handle
    clobbered = set()
    for stmt in block:
        if isinstance(stmt, ir.InPlaceOp):
            if isinstance(stmt.target, ir.NameRef):
                if stmt.target in clobbered:
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
        assert isinstance(node.target, ir.NameRef)
        t = self.symbols.check_type(node.target)
        target = self.symbols.make_unique_name_like(target, t)
        self.current[stmt.target] = target
        return target


def rename_intermediate(symbols: SymbolTable, always_rename: Set[ir.NameRef], block: Iterable[ir.StmtBase]):
    """
    Renames clobbering assignments to cases where the variable name is ephemeral with more than one global assignment
    as determined by always_rename.

    :param symbols:
    :param always_rename:
    :param block:
    :return:
    """
    block = list(block)
    counts = get_block_assign_counts(block)
    curr_assign_index = Counter()
    rewriter = ExprRewriter(symbols)
    repl = []
    for stmt in block:
        if isinstance(stmt, ir.Assign):
            value = rewriter.rewrite_expr(stmt.value)
            target = stmt.target
            if isinstance(target, ir.NameRef):
                curr_assign_index[target] += 1
                index = curr_assign_index[target]
                if index < counts[target] or target in always_rename:
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


def optimize_statements(node: ir.Function, symbols: SymbolTable):
    """
    Function scope dead code removal pass
    :param node:
    :param symbols:
    :return:
    """
    # first expand in place assignments where possible
    assign_expander = BlockRewriter(expand_assigns)
    body = assign_expander.visit(node.body)
    # now remove any dead assignments
    ephemeral = find_ephemeral_references(body, symbols)
    ephemeral.difference_update(node.args)
    # remove any unreferenced assignments
    dce = BlockRewriter(partial(dead_assign_elim, ephemeral))
    body = dce.visit(body)
    # rename intermediate references,
    always_rename = find_ephemeral_references(body, symbols)
    # We can ignore renamining of any references that are only assigned once
    always_rename.difference_update(get_uniquely_assigned(body))
    always_rename.difference_update(node.args)
    block_renamer = partial(rename_intermediate, symbols, always_rename)
    renamer = BlockRewriter(block_renamer)
    body = renamer.visit(body)

    return ir.Function(node.name, node.args, body)
