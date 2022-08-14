import operator

from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import npmd.ir as ir
from npmd.blocks import build_function_graph, BasicBlock, FlowGraph
from npmd.traversal import walk_nodes


def inline_const_branches(stmts: List[ir.StmtBase]):
    repl = []
    for stmt in stmts:
        if isinstance(stmt, ir.IfElse):
            if isinstance(stmt.test, ir.CONSTANT):
                if operator.truth(stmt.test):
                    live_branch = inline_const_branches(stmt.if_branch)
                    repl.extend(live_branch)
                else:
                    live_branch = inline_const_branches(stmt.else_branch)
                    repl.extend(live_branch)
            else:
                stmt.if_branch = inline_const_branches(stmt.if_branch)
                stmt.else_branch = inline_const_branches(stmt.else_branch)
                repl.append(stmt)
        else:
            if isinstance(stmt, (ir.ForLoop, ir.WhileLoop)):
                stmt.body = inline_const_branches(stmt.body)
            repl.append(stmt)
    return repl


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
    # Todo: gather first.. how did this ever work?
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
    # Since the graph represents a view, we want to avoid
    # having to clear one side of an if statement, if the branch remains intact,
    # since this has a tendency to result in phantom blocks.
    for block in graph.nodes():
        if block.is_branch_block:
            assert not isinstance(block.first.test, ir.CONSTANT)
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
