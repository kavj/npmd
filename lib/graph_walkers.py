import itertools

import networkx as nx
import lib.ir as ir

from typing import Generator, Iterator, List, Optional

from blocks import BasicBlock, FunctionContext, dominator_tree


def insert_block_before(func: FunctionContext, block: BasicBlock, statements: List[ir.Statement]):
    reduced = get_reduced_graph(func)
    parents = [p for p in func.graph.predecessors(block) if (p, block) in reduced.edges()]
    for p in parents:
        func.graph.remove_edge(p, block)
    func.add_block(statements, block.depth, parents, [block])


def get_branch_entry_points(func: FunctionContext, branch_block: BasicBlock):
    assert branch_block.is_branch_block
    if_branch = None
    else_branch = None
    branch = branch_block.first
    for s in func.graph.successors(branch):
        if branch.if_branch == s.label:
            if_branch = s
        elif branch.else_branch == s.label:
            else_branch = s
    return if_branch, else_branch


def get_loop_entry_block(func: FunctionContext, loop_block: BasicBlock) -> Optional[BasicBlock]:
    assert loop_block.is_loop_block
    for s in func.graph.successors(loop_block):
        if s.depth == loop_block.depth + 1:
            return s


def get_loop_exit_block(func: FunctionContext, loop_block: BasicBlock) -> Optional[BasicBlock]:
    assert loop_block.is_loop_block
    for s in func.graph.successors(loop_block):
        if s.depth == loop_block.depth:
            return s


def get_blocks_in_loop(graph: nx.DiGraph, header: BasicBlock):
    """
    Get blocks including divergent components
    :param graph:
    :param header:
    :return:
    """
    assert header.is_loop_block

    visited = set()
    queued = [s for s in graph.successors(header) if s.depth == header.depth + 1]
    while queued:
        block = queued.pop()
        if block not in visited:
            # can be queued more than once
            visited.add(block)
            queued.extend(s for s in graph.successors(block) if s not in visited and s.depth == block.depth)
    return visited


def find_loop_exit(func: FunctionContext, header: BasicBlock):
    """
    This is meant to grab loop exits, which include cases where we must pass through a nested loop to escape.
    :param func:
    :param header:
    :return:
    """
    doms = dominator_tree(func)
    for block in nx.dfs_preorder_nodes(doms, header):
        if block.depth == header.depth:
            return block

def get_reduced_graph(func: FunctionContext, doms: Optional[nx.DiGraph] = None) -> nx.DiGraph:
    if doms is None:
        doms = dominator_tree(func)
    back_edges = set()
    for u in nx.dfs_preorder_nodes(func.graph, func.entry_point):
        for v in func.graph.successors(u):
            if v in nx.ancestors(doms, u):
                back_edges.add((u, v))
    return nx.edge_subgraph(func.graph, [e for e in func.graph.edges() if e not in back_edges])


def walk_graph(graph: nx.DiGraph) -> Iterator[BasicBlock]:
    return iter(graph.nodes())


def find_branch_exit(func: FunctionContext, header: BasicBlock):
    assert header.is_branch_block
    doms = dominator_tree(func)
    entry_blocks = list(func.graph.successors(header))
    header_descendants = nx.descendants(doms, header)
    branch_descendants = set()
    for e in entry_blocks:
        branch_descendants.add(e)
        branch_descendants.update(nx.descendants(doms, e))
    # This probably needs to skip break nodes
    for block in nx.dfs_preorder_nodes(doms, header):
        if block is not header and block in header_descendants and block not in branch_descendants:
            return block


def get_reachable_blocks(graph: nx.DiGraph, entry_point: BasicBlock):
    return nx.descendants(graph, entry_point)


def get_reachable_nodes(graph: nx.DiGraph, entry_point: BasicBlock) -> Generator[ir.Statement, None, None]:
    reachable_blocks = get_reachable_blocks(graph, entry_point)
    yield from itertools.chain(*reachable_blocks)


def create_graph_view(func: FunctionContext, include_back_edges: bool = False, include_terminal_edges: bool = False):
    if include_back_edges:
        if include_terminal_edges:
            return func.graph
        else:
            return nx.subgraph_view(func.graph, filter_edge=lambda e: e[0].unterminated)
    else:
        dominators = dominator_tree(func)
        if include_terminal_edges:
            return nx.subgraph_view(func.graph, filter_edge=lambda e: e[1] not in nx.ancestors(dominators, e[0]))
        else:
            return nx.subgraph_view(func.graph, filter_edge=lambda e: e[0].unterminated and e[1] not in nx.ancestors(dominators, e[0]))


def preorder_walk(graph: nx.DiGraph, entry_point: Optional[BasicBlock] = None):
    return nx.dfs_preorder_nodes(graph, entry_point)


def postorder_walk(graph: nx.DiGraph, entry_point: Optional[BasicBlock] = None):
    return nx.dfs_postorder_nodes(graph, entry_point)
