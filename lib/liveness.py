from __future__ import annotations
import itertools
import operator

from collections import Counter, defaultdict, deque
from functools import singledispatch, singledispatchmethod
from typing import Dict, Iterable, Set

import lib.ir as ir

from lib.blocks import BasicBlock, FlowGraph, build_function_graph, dominator_tree
from lib.errors import CompilerError
from lib.symbol_table import SymbolTable
from lib.traversal import get_statement_lists, walk_parameters
from lib.utils import is_entry_point, unpack_iterated


# Most of this work ended up being done elsewhere in controlflow now. This can now do slightly more precise liveness
# The intention here is to better understand what assignments may not escape and can therefore be localized or removed
# if dead.


@singledispatch
def get_assigned(node):
    raise TypeError()


@get_assigned.register
def _(node: ir.StmtBase):
    return  # avoid yielding None
    yield


@get_assigned.register
def _(node: ir.Assign):
    for p in walk_parameters(node.value):
        if isinstance(p, ir.NameRef):
            yield p


@get_assigned.register
def _(node: ir.ForLoop):
    for target, value in unpack_iterated(node):
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
    for target, value in unpack_iterated(node):
        yield from walk_parameters(value)
        if not isinstance(target, ir.NameRef):
            yield from walk_parameters(target)


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


def find_live_in_out(graph: FlowGraph) -> Dict[BasicBlock, BlockLiveness]:
    block_liveness = {}
    for node in graph.nodes():
        kills = set()
        referenced = set()
        live_in = set()
        for stmt in node:
            for name in get_referenced(stmt):
                if name not in kills:
                    live_in.add(name)
                referenced.add(name)
            kills.update(get_assigned(stmt))
        single_block_liveness = BlockLiveness(live_in, kills)
        block_liveness[node] = single_block_liveness
    changed = True
    # mark basic live
    while changed:
        changed = False
        for node in graph.nodes():
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


def check_all_assigned(graph: FlowGraph):
    entry = graph.entry_block
    visited = {entry}
    bound_after = {entry: set(entry.first.args)}
    doms = dominator_tree(graph)
    # Todo: cascade assign shows evidence of a bug here in CFG generation... investigate
    # prune back edges
    maybe_unbound = set()

    body_entry, = graph.successors(entry)
    queued = deque()
    queued.append(body_entry)

    def update_unbound(expr: ir.ValueRef, safe: Set[ir.NameRef]):
        for p in walk_parameters(expr):
            if p not in safe:
                maybe_unbound.add(p)

    while queued:
        block = queued.popleft()
        assigned = set.intersection(*(bound_after[p]
                                      for p in graph.predecessors(block)
                                      if p in visited))
        for stmt in block:
            if isinstance(stmt, ir.Assign):
                update_unbound(stmt.value, assigned)
                if isinstance(stmt.target, ir.NameRef):
                    assigned.add(stmt.target)
                else:
                    update_unbound(stmt.target, assigned)
            elif isinstance(stmt, ir.InPlaceOp):
                for p in walk_parameters(stmt.value):
                    if p not in assigned:
                        maybe_unbound.add(p)
            elif isinstance(stmt, ir.ForLoop):
                for target, value in unpack_iterated(stmt):
                    update_unbound(value, assigned)
                    if isinstance(target, ir.NameRef):
                        assigned.add(target)
                    else:
                        update_unbound(target, assigned)
            elif isinstance(stmt, (ir.IfElse, ir.WhileLoop)):
                update_unbound(stmt.test, assigned)
            elif isinstance(stmt, (ir.SingleExpr, ir.Return)):
                update_unbound(stmt.value, assigned)

        visited.add(block)
        for succ in graph.successors(block):
            if succ not in visited:
                queued.append(succ)
        bound_after[block] = assigned
    return visited


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


def drop_unused_symbols(func: ir.Function, symbols: SymbolTable):
    """
    Purges any dead symbols from the symbol table. This is useful to remove non-ephemeral variables
    that are simply unused, so that they are not declared.
    :param func:
    :param symbols:
    :return:
    """
    assigned = set()
    referenced = set(func.args)
    for stmts in get_statement_lists(func):
        for stmt in stmts:
            assigned.update(get_assigned(stmt))
            referenced.update(get_referenced(stmt))
    dead_names = set()
    for sym_name in symbols.all_locals:
        if sym_name not in referenced and sym_name not in assigned:
            dead_names.add(sym_name)
    for name in dead_names:
        symbols.drop_symbol(name)


def get_unreachable_blocks(graph: FlowGraph):
    # Note: we can't just look for in-degree 0, since we can have unreachable loops
    # This just axes anything that isn't
    entry_block = graph.entry_block
    # Todo: might be an issue here.. investigate later..
    assert entry_block is not None
    unreachable = set(graph.nodes())
    unreachable.difference_update(graph.reachable_nodes())
    return unreachable


def group_by_statement_list(blocks: Iterable[BasicBlock]):
    by_stmt_list = defaultdict(set)
    id_to_list = {}
    for block in blocks:
        statements = block.statements
        list_id = id(statements)
        by_stmt_list[list_id].add(block)
        id_to_list[list_id] = statements
    return by_stmt_list, id_to_list


def get_loop_exit(graph: FlowGraph, node: BasicBlock):
    assert node.is_loop_block
    for succ in graph.successors(node):
        if succ.depth == node.depth:
            return succ


def get_loop_body(graph: FlowGraph, node: BasicBlock):
    assert node.is_loop_block
    for succ in graph.successors(node):
        if succ.depth == node.depth + 1:
            return succ


def remove_dead_edges(graph: FlowGraph):
    nodes_with_dead_edges = set()
    for node in graph.nodes():
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
            if_block, else_block = graph.get_branch_entry_points(node)
            if operator.truth(test) and graph.graph.has_edge(node, else_block):
                graph.remove_edge(node, else_block)
            elif graph.graph.has_edge(node, if_block):
                graph.remove_edge(node, if_block)
        else:
            header = node.first
            assert isinstance(header, ir.WhileLoop) and isinstance(header.test, ir.CONSTANT)
            if operator.truth(header.test):
                exit_block = get_loop_exit(graph, node)
                if graph.graph.has_edge(node, exit_block):
                    graph.remove_edge(node, exit_block)
            else:
                entry_block = get_loop_body(graph, node)
                if graph.graph.has_edge(node, entry_block):
                    graph.remove_edge(node, entry_block)


def remove_trivial_continues(node: ir.Function):
    for stmts in get_statement_lists(node):
        for stmt in stmts:
            if isinstance(stmt, (ir.ForLoop, ir.WhileLoop)):
                while stmt.body and isinstance(stmt.body[-1], ir.Continue):
                    stmt.body.pop()


def remove_statements_following_terminals(node: ir.Function, symbols: SymbolTable):
    """
     Removes any statements following a break, continue, or return statement in any statement list
     :param symbols:
     :param node:
     :return:
     """
    for stmt_list in get_statement_lists(node):
        for i, stmt in enumerate(stmt_list):
            if isinstance(stmt, (ir.Break, ir.Continue, ir.Return)):
                for j in range(len(stmt_list) - i - 1):
                    stmt_list.pop()
                break
    drop_unused_symbols(node, symbols)


def remove_unreachable_blocks(graph: FlowGraph):
    # Since the graph represents a view, we want to avoid
    # having to clear one side of an if statement, if the branch remains intact,
    # since this has a tendency to result in phantom blocks.
    for block in graph.nodes():
        if block.is_branch_block:
            if isinstance(block.first.test, ir.CONSTANT):
                # These lead to extremely misleading control flow graphs, because the statements end up being replaced
                # by an empty block due to the prescence of an ifelse node, rather than a simplified path.
                msg = f'Remove unreachable blocks expects static constant branches to be removed. Received {block.first.test}, position: {block.first.pos}.'
                raise CompilerError(msg)
    remove_dead_edges(graph)
    # get all unreachable nodes
    entry_block = graph.entry_block
    assert entry_block is not None
    func = entry_block.first
    unreachable = get_unreachable_blocks(graph)
    unreachable_grouped, unreachable_id_to_list = group_by_statement_list(unreachable)
    grouped, id_to_list = group_by_statement_list(graph.nodes())
    # Now find statement lists that are completely unreachable

    for list_id, unreachable_blocks in unreachable_grouped.items():
        # get total blocks
        total_blocks = grouped[list_id]
        reachable_blocks = total_blocks.difference(unreachable_blocks)
        stmts = unreachable_id_to_list.pop(list_id)
        if reachable_blocks:
            # rebuild  from remaininive blocks, then generate the original list
            repl = []
            for block in sorted(reachable_blocks, key=lambda b: b.start):
                repl.extend(block)
            stmts.clear()
            stmts.extend(repl)
        else:
            # mitigates hidden bugs, since these really are gone from the list
            stmts.clear()
    remove_trivial_continues(func)
    repl_graph = build_function_graph(func)
    # can be added better later..
    remove_dead_edges(repl_graph)
    unreachable = get_unreachable_blocks(repl_graph)
    for u in unreachable:
        repl_graph.graph.remove_node(u)

    return repl_graph


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


def remove_dead_statements(node: ir.Function, symbols: SymbolTable):
    graph = build_function_graph(node)
    liveness = find_live_in_out(graph)
    dead = mark_dead_statements(liveness, symbols)
    repl = []
    for stmts in get_statement_lists(node):
        changed = False
        for stmt in stmts:
            if is_entry_point(stmt) or stmt not in dead:
                repl.append(stmt)
            else:
                changed = True
        if changed:
            stmts.clear()
            stmts.extend(repl)
        repl.clear()
    drop_unused_symbols(node, symbols)


def remove_unreachable_statements(func: ir.Function, symbols: SymbolTable):
    graph = build_function_graph(func)
    remove_unreachable_blocks(graph)
    remove_dead_statements(func, symbols)
    remove_statements_following_terminals(func, symbols)
