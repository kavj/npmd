from __future__ import annotations
import itertools
from collections import defaultdict
from functools import singledispatchmethod
import networkx as nx
from analysis import get_names, get_read_and_assigned
from blocks import BasicBlock, build_function_graph, get_blocks_in_loop, get_loop_header_block
from errors import CompilerError
from symbol_table import SymbolTable
from traversal import walk_parameters
from utils import is_entry_point, unpack_iterated
from typing import Dict, Iterable, List, Set, Union

import ir


# Most of this work ended up being done elsewhere in controlflow now. This can now do slightly more precise liveness
# The intention here is to better understand what assignments may not escape and can therefore be localized or removed
# if dead.


def drop_unused_symbols(stmts: Iterable[ir.StmtBase], args: Iterable[ir.NameRef], symbols: SymbolTable):
    """
    Purges any dead symbols from the symbol table. This is useful to remove non-ephemeral variables
    that are simply unused, so that they are not declared.
    :param stmts:
    :param args:
    :param symbols:
    :return:
    """
    # Todo: this doesn't capture referenced... how did it ever work?
    # stmt_list = list(stmts)
    referenced = get_names(stmts)
    referenced.update(args)
    dead_names = set()
    for sym_name in symbols.all_locals:
        if sym_name not in referenced:
            dead_names.add(sym_name)
    for name in dead_names:
        symbols.drop_symbol(name)


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
        elif is_entry_point(stmt):
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


def propagate_liveness(graph: nx.DiGraph, block_liveness: Dict[BasicBlock, BlockLiveness]):
    changed = True
    # mark basic live
    while changed:
        changed = False
        for node in graph.nodes():
            single_block_liveness = block_liveness[node]
            succ_liveness = (block_liveness[s] for s in graph.successors(node))
            changed |= single_block_liveness.update_liveness(succ_liveness)


def find_live_in_out(graph: nx.DiGraph) -> Dict[BasicBlock, BlockLiveness]:
    block_liveness = {}
    for node in graph.nodes():
        kills, referenced, live_in = get_read_and_assigned(node)
        single_block_liveness = BlockLiveness(live_in, kills)
        block_liveness[node] = single_block_liveness
    propagate_liveness(graph, block_liveness)
    return block_liveness


def get_clobbers(blocks: Iterable[BasicBlock]):
    for stmt in itertools.chain(*blocks):
        if isinstance(stmt, (ir.Assign, ir.InPlaceOp)):
            if isinstance(stmt.target, ir.NameRef):
                yield stmt.target
        elif isinstance(stmt, ir.ForLoop):
            # nested loop should not clobber
            for target, _ in unpack_iterated(stmt.target, stmt.iterable):
                if isinstance(target, ir.NameRef):
                    yield target


def find_loop_iterables_clobbered_by_body(graph: nx.DiGraph, node: Union[ir.ForLoop, BasicBlock]):
    """

    :param graph:
    :param node:
    :return:
    """

    # Todo: we'll need a check for array augmentation as well later to see whether it's easily provable
    # that writes match the read pattern at entry
    header = node if isinstance(node, BasicBlock) else get_loop_header_block(graph, node)
    blocks = set(get_blocks_in_loop(graph, header, include_header=False))
    assigned = set(get_clobbers(blocks))
    loop_header_stmt = header.first
    clobbered = set()
    for _, iterable in unpack_iterated(loop_header_stmt.target, loop_header_stmt.iterable):
        # This is used so that we can rename clobbered names before processing the loop header.
        # generally they shouldn't come up in the first place
        for p in walk_parameters(iterable):
            if p in assigned:
                clobbered.add(p)
    return clobbered


def find_loop_local_liveness(graph: nx.DiGraph, node: ir.ForLoop):
    """
    :param graph:
    :param node:
    :return:
    """

    header = node if isinstance(node, BasicBlock) else get_loop_header_block(graph, node)
    # To avoid complicating analysis, ensure nothing declared in the header is clobbered in the body
    clobbered_iterables = find_loop_iterables_clobbered_by_body(graph, header)
    if clobbered_iterables:
        clobber_str = ", ".join(c.name for c in clobbered_iterables)
        msg = f'The following names are iterated over and clobbered by the loop body: "{clobber_str}"'
        raise ValueError(msg)
    blocks = set(get_blocks_in_loop(graph, header))
    blocks.add(header)
    loop_graph = graph.subgraph(blocks)
    block_liveness = {}
    clobbers = get_clobbers(loop_graph.nodes())
    for node in loop_graph.nodes():
        kills, referenced, live_in = get_read_and_assigned(node)
        refs = kills.union(referenced)
        refs.update(live_in)
        for r in refs:
            if r not in clobbers:
                kills.discard(r)
                referenced.discard(r)
                live_in.discard(r)
        single_block_liveness = BlockLiveness(live_in, kills)
        block_liveness[node] = single_block_liveness
    # Now mark anything that could be kept alive by a live alias
    return block_liveness


def check_all_assigned(graph: nx.DiGraph):
    liveness = find_live_in_out(graph.graph)
    entry = graph.entry_block
    visited = {entry}
    bound_after = {entry: {*entry.first.args}}
    # Todo: cascade assign shows evidence of a bug here in CFG generation... investigate
    body_entry, = graph.successors(entry)
    for block in nx.dfs_preorder_nodes(graph.graph, body_entry):
        # note: this wouldn't be sufficient in the presence of irreducible control flow
        liveinfo = liveness[block]
        # check if bound along all paths
        bound_by = set.intersection(*(bound_after[p] for p in graph.predecessors(block) if p in visited))
        maybe_unbound = set()
        for name in liveinfo.live_in:
            if name not in bound_by:
                maybe_unbound.add(name)
        if maybe_unbound:
            s = ', '.join(name.name for name in maybe_unbound)
            msg = f'Names: "{s}" do not reach along all paths'
            raise CompilerError(msg)
        bound_by.update(liveinfo.kills)
        bound_after[block] = bound_by
        visited.add(block)


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


def mark_dead_statements(block_to_liveness: Dict[BasicBlock, BlockLiveness], symbols: SymbolTable):
    """
        control flow graph is used as an analysis too only
        :param block_to_liveness:
        :param symbols:
        :return:
    """

    dead = set()

    for block, livenesss_info in block_to_liveness.items():
        if block.is_loop_block or block.is_branch_block or block.is_entry_block:
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
                dead.add(stmt)
    return dead


def remove_statements_from(body: List[ir.Statement], dead: Iterable[ir.Statement]):
    repl = []
    for stmt in body:
        if isinstance(stmt, ir.IfElse):
            if_branch = remove_statements_from(stmt.if_branch, dead) if stmt.if_branch else []
            else_branch = remove_statements_from(stmt.else_branch, dead) if stmt.else_branch else []
            stmt = ir.IfElse(stmt.test, if_branch, else_branch, stmt.pos)
            if if_branch or else_branch:
                repl.append(stmt)
        elif isinstance(stmt, ir.ForLoop):
            body = remove_statements_from(stmt.body, dead)
            stmt = ir.ForLoop(stmt.target, stmt.iterable, body, stmt.pos)
            repl.append(stmt)
        elif isinstance(stmt, ir.WhileLoop):
            body = remove_statements_from(stmt.body, dead)
            stmt = ir.WhileLoop(stmt.test, body, stmt.pos)
            repl.append(stmt)
        elif stmt not in dead:
            repl.append(stmt)
    return repl


def remove_dead_statements(func: ir.Function, symbols: SymbolTable):
    graph = build_function_graph(func)
    liveness = find_live_in_out(graph)
    dead = mark_dead_statements(liveness, symbols)
    body = remove_statements_from(func.body, dead)
    func = ir.Function(func.name, func.args, body)
    graph = build_function_graph(func)
    stmts = itertools.chain(*(block for block in graph.nodes()))
    drop_unused_symbols(stmts, func.args, symbols)
    return func
