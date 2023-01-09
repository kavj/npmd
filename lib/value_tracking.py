import networkx as nx

import lib.ir as ir
from lib.blocks import BasicBlock, dominator_tree, FlowGraph, get_blocks_in_loop
from lib.liveness import find_live_in_out
from collections import defaultdict, deque
from typing import DefaultDict

from lib.utils import unpack_iterated


def collect_reaching(graph: nx.DiGraph, start: BasicBlock, latest: DefaultDict[BasicBlock, DefaultDict], var: ir.NameRef):
    """
    This assumes no back edges..

    :param graph:
    :param start:
    :param latest:
    :param var:
    :return:
    """
    queued = deque(graph.predecessors(start))
    seen = set()
    defs = set()
    pending = []
    while queued:
        block = queued.pop()
        if block not in seen:
            value = latest[block].get(var)
            if value is not None:
                defs.add(value)


def get_last_assign(block: BasicBlock):
    if block.is_loop_block:
        header = block.first
        if isinstance(header, ir.ForLoop):
            return {target: value  for (target, value) in unpack_iterated(header)}
        else:
            latest = {}
            for stmt in block:
                if isinstance(stmt, ir.Assign) and isinstance(stmt.target, ir.NameRef):
                    latest[stmt.target] = stmt
            return latest


def track_loop(graph: FlowGraph, entry: BasicBlock):
    """
    based on Braun, et al., Simple and Efficient Construction of Static Single Assignment form, Section 3.3

    :param graph:
    :param entry:
    :return:
    """
    assert entry.is_loop_block
    doms = dominator_tree(graph)
    loop_graph = get_blocks_in_loop(graph, entry).copy()
    back_edges = [(source, sink) for source, sink in loop_graph.edges() if sink in doms.ancestors(source)]
    loop_graph.remove_edges_from(back_edges)
    liveness = find_live_in_out(graph)
    assert entry in loop_graph.nodes()
    assigns = {}
    for block in nx.topological_sort(loop_graph):
        assigns[block] = get_last_assign(block)

    all_assigned = set()
    for block in nx.topological_sort(loop_graph):
        if block.is_loop_block and isinstance(block.first, ir.ForLoop):
            for target, iterable in unpack_iterated(block.first):
                all_assigned.add(target)
        else:
            for stmt in block:
                if isinstance(stmt, ir.Assign) and isinstance(stmt.target, ir.NameRef):
                    all_assigned.add(stmt.target)

    live_across_latch = liveness[entry].live_in.intersection(all_assigned)
    latest = defaultdict(lambda x: defaultdict(set))
    # now check live in
    for block in nx.topological_sort(loop_graph):
        if block is entry:
            continue
        block_live = liveness[block]
        for pred in loop_graph.predecessors(block):
            for term in block_live.live_in:
                latest[block][term].update(latest[pred][term])

