import operator

import networkx as nx

import lib.ir as ir

from dataclasses import fields
from typing import Iterable
from lib.blocks import FunctionContext, dominator_tree
from lib.graph_walkers import walk_graph


def statements_match(*stmts: Iterable[ir.StmtBase]):
    first = stmts[0]
    first_type = type(first)
    fields_first = fields(first)
    assert isinstance(first, ir.StmtBase)
    for stmt in stmts:
        if type(stmt) != first_type:
            return False
        if fields_first != fields(stmt):
            raise ValueError(f'field names do not match between "{first}" and "{stmt}"')
        # We have matching type and fields (second should always be true unless live objects updated).
        # Now we need to check if values other than positional information all match.
        # We're avoiding asdict here to avoid the use of deepcopy
        for f in fields_first:
            name = f.name
            if f.name == 'pos':
                continue
            if getattr(first, name) != getattr(stmt, name):
                return False
    return True


def hoist_terminals(func: FunctionContext):
    """
    Try to turn things like

    if value:
       ...
       continue
    else:
       ...
       continue

    into

    if value:
       ...
    else:
       ...
    continue

    :param func:
    :return:
    """
    dominators = dominator_tree(func)
    # we aren't going to try to do this across weird loop structures, so we can remove break edges
    for block in nx.dfs_postorder_nodes(dominators, func.entry_point):
        if block.is_branch_block:
            terminal_descendants = [d for d in nx.descendants(dominators, block) if d.terminated]
            if len(terminal_descendants) == 2:
                u, v = terminal_descendants
                if u.terminated and v.terminated and u.depth == v.depth == block.depth:
                    if statements_match(u.last, v.last):
                        # verify that targets match
                        u_succ, = func.graph.successors(u)
                        v_succ, = func.graph.successors(v)
                        if u_succ is v_succ:
                            # we have a jump to the same block
                            u_last = u.pop_statement()
                            v_last = v.pop_statement()
                            terminal = u_last if u_last.pos > v_last.pos else v_last
                            for b in (u, v):
                                func.graph.remove_edge(b, u_succ)
                            func.add_block([terminal], u.depth, [u, v], [u_succ])


def inline_const_branches(graph: nx.DiGraph):
    """
    Remove any constant branch nodes and link their predecessors to whatever branch remains live
    :param graph:
    :return:
    """
    inlinable_branches = [block for block in walk_graph(graph) if block.is_branch_block
                          and isinstance(block.first.test, ir.CONSTANT)]
    for branch in inlinable_branches:
        header: ir.IfElse = branch.first
        live_block_label = header.if_branch if operator.truth(header.test) else header.else_branch
        live_block = graph[live_block_label]
        for p in graph.predecessors(header):
            graph.add_edge(p, live_block)
        graph.remove_node(branch)
