from __future__ import annotations
import itertools
import operator

from collections import Counter, defaultdict
from functools import singledispatchmethod
from typing import Dict, Iterable, Set

import networkx as nx

import lib.ir as ir

from lib.statement_utils import get_assigned, get_expressions, get_referenced
from lib.blocks import BasicBlock, FunctionContext
from lib.graph_walkers import get_branch_entry_points, get_reachable_blocks,  get_loop_entry_block, \
    get_loop_exit_block,get_reduced_graph, walk_graph
from lib.symbol_table import SymbolTable
from lib.expression_walkers import walk_parameters


# Most of this work ended up being done elsewhere in controlflow now. This can now do slightly more precise liveness
# The intention here is to better understand what assignments may not escape and can therefore be localized or removed
# if dead.

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

    def mark_link(self, expr: ir.ValueRef, link_to: ir.Assign):
        for param in walk_parameters(expr):
            latest_param_assign = self.latest.get(param)
            if latest_param_assign is not None and latest_param_assign not in self.live:
                self.linked[link_to].add(latest_param_assign)

    @singledispatchmethod
    def mark(self, stmt: ir.StmtBase):
        if not isinstance(stmt, ir.StmtBase):
            msg = f'Not a statement "{stmt}"'
            raise TypeError(msg)
        elif isinstance(stmt, (ir.ForLoop, ir.WhileLoop, ir.IfElse)):
            msg = f'Cannot mark on entry point "{stmt}"'
            raise TypeError(msg)
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
    def _(self, stmt: ir.SingleExpr):
        self.mark_live(stmt.value)
        self.live.add(stmt)

    @mark.register
    def _(self, node: ir.Return):
        if node.value is not None:
            self.mark_live(node.value)
        self.live.add(node)


class BlockLiveness:

    def __init__(self, live_in: Set[ir.NameRef], kills: Set[ir.NameRef]):
        self.live_in = live_in
        self.kills = kills
        self.live_out = set()

    def discard_entry(self, name: ir.NameRef):
        self.live_in.discard(name)
        self.live_out.discard(name)
        self.kills.discard(name)

    def entries(self):
        yield from itertools.chain(self.live_in, self.kills)

    def update_liveness(self, successors: Iterable[BlockLiveness]):
        successors_live_in = set()
        for s in successors:
            successors_live_in.update(s.live_in)
        live_in_count = len(self.live_in)
        live_out_count = len(self.live_out)
        self.live_out.update(successors_live_in)
        pass_through = successors_live_in.difference(self.kills)
        self.live_in.update(pass_through)
        changed = live_in_count != len(self.live_in) or live_out_count != len(self.live_out)
        return changed


def find_live_in_out(graph: nx.DiGraph, tracked: Set[ir.NameRef]) -> Dict[BasicBlock, BlockLiveness]:
    block_liveness = {}
    for node in walk_graph(graph):
        kills = set()
        referenced = set()
        live_in = set()
        for stmt in node:
            for name in get_referenced(stmt):
                if name not in kills and name in tracked:
                    live_in.add(name)
                referenced.add(name)
            kills.update(name for name in get_assigned(stmt) if name in tracked)
        single_block_liveness = BlockLiveness(live_in, kills)
        block_liveness[node] = single_block_liveness
    changed = True
    # mark basic live
    while changed:
        changed = False
        for node in walk_graph(graph):
            single_block_liveness = block_liveness[node]
            succ_liveness = (block_liveness[s] for s in graph.successors(node))
            changed |= single_block_liveness.update_liveness(succ_liveness)
    return block_liveness


def find_ephemeral_assigns(liveness: Dict[BasicBlock, BlockLiveness]) -> Set[ir.Assign]:
    """
    :param liveness:
    :return:
    """
    ephemeral = set()
    for block, liveness_info in liveness.items():
        # find assign counts by name
        counts = Counter()
        for stmt in block:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.target, ir.NameRef):
                # pure clobber
                counts[stmt.target] += 1
        seen_counts = Counter()
        for stmt in block:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.target, ir.NameRef):
                seen_counts[stmt.target] += 1
                if seen_counts[stmt.target] < counts[stmt.target]:
                    # intermediate assignment
                    ephemeral.add(stmt)
                elif stmt.target not in liveness_info.live_in and stmt.target not in liveness_info.live_out:
                    ephemeral.add(stmt)
    return ephemeral


def check_for_maybe_unbound_names(func: FunctionContext):
    liveness = find_live_in_out(func.graph, set(func.symbols.all_locals))
    assigned_after = {func.entry_point: func.args}
    unbound = {}

    no_backedge_graph = get_reduced_graph(func)
    for block in nx.topological_sort(no_backedge_graph):
        if block.is_function_entry:
            continue
        block_liveness = liveness[block]
        # get everything we know is bound after this
        bound = set(block_liveness.kills)
        for p in no_backedge_graph.predecessors(block):
            bound.update(assigned_after[p])
        assigned_after[block] = bound

    # Now check if each block has any live in name references that are not guaranteed to be bound
    # on exit by all predecessors
    for block in get_reachable_blocks(no_backedge_graph, func.entry_point):
        block_liveness = liveness[block]
        if not block_liveness.live_in:
            continue
        bound_on_entry = set()
        for p in no_backedge_graph.predecessors(block):
            bound_on_entry.update(assigned_after[p])
        if not bound_on_entry.issuperset(block_liveness.live_in):
            unbound[block] = {name for name in block_liveness.live_in if name not in bound_on_entry}
    return unbound, assigned_after


def dump_live_info(info: Dict[BasicBlock, BlockLiveness], ignores=()):
    for block, liveness in info.items():
        print(f'block {block.label}')
        # ignores are to avoid looking at things like arguments
        live_ins = ', '.join(v.name for v in liveness.live_in if v not in ignores)
        live_outs = ', '.join(v.name for v in liveness.live_out if v not in ignores)
        kills = ', '.join(v.name for v in liveness.kills if v not in ignores)
        print(f'live in:\n    {live_ins}')
        print(f'live outs:\n    {live_outs}')
        print(f'kills:\n    {kills}')


def drop_unused_symbols(func: FunctionContext):
    """
    Purges any dead symbols from the symbol table. This is useful to remove non-ephemeral variables
    that are simply unused, so that they are not declared.
    :param func:
    :return:
    """
    assigned = set()
    referenced = set(func.entry_point.first.args)
    symbols = func.symbols
    for block in get_reachable_blocks(func.graph, func.entry_point):
        for stmt in block:
            assigned.update(get_assigned(stmt))
            referenced.update(get_referenced(stmt))
    dead_names = set()
    for sym_name in symbols.all_locals:
        if sym_name not in referenced and sym_name not in assigned:
            dead_names.add(sym_name)
    for name in dead_names:
        symbols.drop_symbol(name)


def remove_dead_edges(func: FunctionContext):
    nodes_with_dead_edges = set()
    graph = func.graph
    for node in get_reachable_blocks(graph, func.entry_point):
        if node.is_branch_block:
            test = node.first.test
            if isinstance(test, ir.CONSTANT):
                nodes_with_dead_edges.add(node)
        elif node.is_loop_block:
            header = node.first
            if isinstance(header, ir.WhileLoop) and isinstance(header.test, ir.CONSTANT):
                nodes_with_dead_edges.add(node)
    for node in nodes_with_dead_edges:
        if node.is_branch_block:
            test = node.first.test
            if_block, else_block = get_branch_entry_points(func, node)
            if operator.truth(test) and graph.has_edge(node, else_block):
                graph.remove_edge(node, else_block)
            elif graph.has_edge(node, if_block):
                graph.remove_edge(node, if_block)
        else:
            header = node.first
            assert isinstance(header, ir.WhileLoop) and isinstance(header.test, ir.CONSTANT)
            if operator.truth(header.test):
                exit_block = get_loop_exit_block(func, node)
                if graph.has_edge(node, exit_block):
                    graph.remove_edge(node, exit_block)
            else:
                entry_block = get_loop_entry_block(func, node)
                if graph.has_edge(node, entry_block):
                    graph.remove_edge(node, entry_block)


def remove_unreachable_blocks(func: FunctionContext):
    reachable_blocks = get_reachable_blocks(func.graph, func.entry_point)
    reachable_blocks.add(func.entry_point)
    removable = [b for b in walk_graph(func.graph) if b not in reachable_blocks]
    func.graph.remove_nodes_from(removable)


def mark_dead_statements(block_to_liveness: Dict[BasicBlock, BlockLiveness], symbols: SymbolTable):
    """
        control flow graph is used as an analysis too only
        :param block_to_liveness:
        :param symbols:
        :return:
    """

    dead = set()

    for block, livenesss_info in block_to_liveness.items():
        if block.is_loop_block or block.is_branch_block or block.is_function_entry:
            continue

        marker = LiveStatementMarker()
        for stmt in block:
            marker.mark(stmt)

        for name in marker.latest:
            if name in livenesss_info.live_out:
                marker.mark_name_live(name)

        for name in symbols.arguments:
            marker.mark_name_live(name)
        # so the problem here is that you have to be cautious

        for stmt in block:
            if stmt not in marker.live:
                # any assignment to a subscript that could alias an input or live value must also be live
                if isinstance(stmt.target, ir.Subscript):
                    if symbols.is_argument(stmt.target.value):
                        continue
                elif isinstance(stmt, ir.InPlaceOp):
                    if symbols.is_argument(stmt.target):
                        continue
                elif any(isinstance(e, ir.Call) for e in get_expressions(stmt)):
                    # this is only safely removable if the call has no side effects
                    continue
                dead.add(stmt)
    return dead


def remove_dead_statements(func: FunctionContext):
    liveness = find_live_in_out(func.graph, set(func.symbols.all_locals))
    dead = mark_dead_statements(liveness, func.symbols)
    for block in get_reachable_blocks(func.graph, func.entry_point):
        changed = False
        if block.is_entry_point:
            continue
        repl = []
        for stmt in block:
            if stmt not in dead:
                repl.append(stmt)
            else:
                changed = True
        if changed:
            block.replace_statements(repl)
    drop_unused_symbols(func)
