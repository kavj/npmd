import itertools

from typing import Iterable

import networkx as nx

import npmd.ir as ir

from npmd.blocks import dominator_tree, FlowGraph
from npmd.traversal import get_statement_lists, walk_nodes, walk_parameters
from npmd.utils import is_entry_point


def find_outermost_branches(graph: FlowGraph):
    """
    Find branches in the form of if-else statements, which does not fall inside any other.
    This ignores loop boundaries, which must be filtered on elsewhere.

    :param graph:
    :return:
    """
    nested = set()
    dominators = dominator_tree(graph)
    branches = []

    for d in nx.dfs_preorder_nodes(dominators, graph.entry_block):
        if d.is_branch_block:
            branch = d.first
            if id(branch) in nested:
                continue
            branches.append(branch)
            for stmt in itertools.chain(walk_nodes(branch.if_branch), walk_nodes(branch.else_branch)):
                if isinstance(stmt, ir.IfElse):
                    nested.add(id(stmt))

    return branches


def group_predicates(predicates: Iterable[ir.ValueRef]):
    """
    Find conditions that are lexically related. This allows us to look for min/max coercions in multi-branch
    cases.

    :return:
    """

    groups = None
    parameter_sets = None
    constants = []  # these are their own group if they exist

    for predicate in predicates:
        params = set(walk_parameters(predicate))
        if isinstance(predicate, ir.CONSTANT):
            constants.append(predicate)
        elif groups:
            for index, (group, param_group_set) in enumerate(zip(groups, parameter_sets)):
                if not params.isdisjoint(param_group_set):
                    param_group_set.update(params)
                    group.append(predicate)
                    break
            else:
                parameter_sets.append(params)
                groups.append([predicate])
        else:
            groups = [[predicate]]
            parameter_sets = [params]
    return groups


def is_perfect_branch_nest(node: ir.IfElse):
    """
    This tests whether branches have the form
    if cond0:
       ...
    elif cond1:
       ...
    ....
    else:
       ...

    where None of these individual branches contain either language level sub-branch or loop statements

    :param node:
    :return:
    """
    queued = [node]
    while queued:
        next_node = queued.pop()
        if any(is_entry_point(stmt) for stmt in walk_nodes(next_node.if_branch)):
            return False
        if len(next_node.else_branch) == 1:
            first = next_node.else_branch[0]
            if is_entry_point(first):
                if isinstance(first, ir.IfElse):
                    queued.append(first)
                else:
                    return False
        else:
            if any(is_entry_point(stmt) for stmt in walk_nodes(next_node.else_branch)):
                return False
    return True


def try_flatten_branch_nest(node: ir.IfElse):
    """
    If this branch is perfectly nested, convert it to a case statement

    Note: this makes further if conversion much simpler
    Note: we may want to break along loop uniform conditions

    :param node:
    :return:
    """

    if not is_perfect_branch_nest(node):
        return node

    seen = set()
    predicates = []
    # Note, we only care about the first true instance of any unique predicate
    # Any duplicates become dead branches, and we need to avoid overwriting their corresponding entries

    queued = [node]
    default_branch = None

    while queued:
        stmt = queued.pop()
        predicate = stmt.test
        if predicate not in seen:
            seen.add(predicate)
            # dead branch
            predicates.append((predicate, stmt.if_branch))
        if stmt.else_branch and is_entry_point(stmt.else_branch[0]):
            queued.append(stmt.else_branch[0])
        else:
            assert not queued
            default_branch = stmt.else_branch

    assert default_branch is not None
    if len(predicates) == 1:
        predicate, branch = predicates.pop()
        return ir.IfElse(predicate, branch, default_branch, node.pos)

    return ir.Case(predicates, default_branch, node.pos)


def flatten_branches(node: ir.Function):
    for stmt_list in get_statement_lists(node):
        for index, stmt in enumerate(stmt_list):
            if isinstance(stmt, ir.IfElse):
                stmt_list[index] = try_flatten_branch_nest(stmt)
