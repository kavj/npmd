import operator
import networkx as nx

from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import ir
from blocks import build_function_graph, BasicBlock, FlowGraph
from traversal import walk_nodes


def get_unreachable_blocks(graph: FlowGraph):
    # Note: we can't just look for in-degree 0, since we can have unreachable loops
    # This just axes anything that isn't
    entry_block = graph.entry_block
    assert entry_block is not None
    unreachable = {n for n in graph.nodes() if n not in nx.descendants(graph.graph, entry_block)}
    return unreachable


def group_by_statement_list(blocks: Iterable[BasicBlock]):
    by_stmt_list = defaultdict(list)
    id_to_list = {}
    for block in blocks:
        statements = block.statements
        list_id = id(statements)
        by_stmt_list[list_id].append(block)
        id_to_list[list_id] = statements
    return by_stmt_list, id_to_list


def build_nesting_map(stmts: List[ir.Statement], existing: Optional[Dict[int, int]] = None, id_to_stmt_list: Optional[Dict[id, List[ir.Statement]]] = None):
    if existing is None:
        existing = defaultdict(list)
        id_to_stmt_list = {}
    list_id = id(stmts)
    id_to_stmt_list[list_id] = stmts
    for stmt in stmts:
        if isinstance(stmt, ir.IfElse):
            existing[list_id].append(id(stmt.if_branch))
            existing[list_id].append(id(stmt.else_branch))
        if isinstance(stmt, (ir.ForLoop, ir.WhileLoop)):
            existing[list_id].append(id(stmt.body))
    return existing, id_to_stmt_list


def remove_dead_edges(graph: FlowGraph):
    for node in graph.nodes():
        if node.is_branch_block:
            test = node.first.test
            if isinstance(test, ir.CONSTANT):
                if_block, else_block = graph.get_branch_entry_points(node)
                if operator.truth(test) and graph.graph.has_edge(node, else_block):
                    graph.remove_edge(node, else_block)
                elif graph.graph.has_edge(node, if_block):
                    graph.remove_edge(node, if_block)
        elif node.is_loop_block:
            header = node.first
            if isinstance(header, ir.WhileLoop) and isinstance(header.test, ir.CONSTANT):
                # this one is common
                if operator.truth(header.test):
                    exit_block = graph.get_loop_exit(node)
                    if graph.graph.has_edge(node, exit_block):
                        graph.remove_edge(node, exit_block)
                else:
                    entry_block = graph.get_loop_body(node)
                    if graph.graph.has_edge(node, entry_block):
                        graph.remove_edge(node, entry_block)


def all_terminated(node: List[ir.Statement]):
    if node:
        last =  node[-1]
        if isinstance(last, (ir.Break, ir.Continue, ir.Return)):
            return True
        elif isinstance(last, ir.IfElse):
            if all_terminated(last.if_branch) and all_terminated(last.else_branch):
                return True
    return False


def get_continue_terminated_branches(node: List[ir.Statement]):
    if node:
        last = node[-1]
        if isinstance(last, ir.Continue):
            yield node
        elif isinstance(last, ir.IfElse):
            yield from get_continue_terminated_branches(last.if_branch)
            yield from get_continue_terminated_branches(last.else_branch)


def strip_continues(node: ir.Function):
    # gather loop nodes
    loop_nodes = [n for n in walk_nodes(node.body) if isinstance(n, (ir.ForLoop, ir.WhileLoop))]
    # first grab trivial
    for node in loop_nodes:
        body = node.body
        if body:
            last = body[-1]
            if isinstance(last, ir.Continue):
                body.pop()
    for node in loop_nodes:
        if all_terminated(node.body):
            terminated = [*get_continue_terminated_branches(node.body)]
            for stmts in terminated:
                assert stmts and isinstance(stmts[-1], ir.Continue)
                stmts.pop()


def remove_trivial_continues(node: ir.Function):
    for node in walk_nodes(node.body):
        if isinstance(node, (ir.ForLoop, ir.WhileLoop)):
            if node.body and isinstance(node.body[-1], ir.Continue):
                node.body.pop()


def remove_statements_following_terminals(node: ir.Function):
    to_trim = [node.body]
    for stmt in walk_nodes(node.body):
        if isinstance(stmt, (ir.ForLoop, ir.WhileLoop)):
            to_trim.append(stmt.body)
        elif isinstance(stmt, ir.IfElse):
            to_trim.append(stmt.if_branch)
            to_trim.append(stmt.else_branch)
    for stmts in to_trim:
        keepcount = 0
        for keepcount, stmt in enumerate(stmts, 1):
            if isinstance(stmt, (ir.Break, ir.Continue, ir.Return)):
                break
        for i in range(len(stmts) - keepcount):
            stmts.pop()


def remove_unreachable_blocks(graph: FlowGraph):
    remove_dead_edges(graph)
    # get all unreachable nodes
    entry_block = graph.entry_block
    assert entry_block is not None
    func = entry_block.first
    unreachable = get_unreachable_blocks(graph)

    grouped, id_to_list = group_by_statement_list(unreachable)
    # Now find statement lists that are completely unreachable

    for list_id, unreachable_blocks in grouped.items():
        # get total blocks
        total_blocks = graph.by_statement_list[list_id]
        reachable_blocks = total_blocks.difference(unreachable_blocks)
        stmts = id_to_list.pop(list_id)

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
    return repl_graph
