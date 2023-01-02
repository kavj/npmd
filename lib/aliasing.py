
"""
Need utilities to check aliasing.

If only certain variable assignments may alias another, we can rename those that don't.
Generally we only care if something is a subscripted target and an alias.


"""

from collections import defaultdict
from typing import Union

import networkx as nx

import lib.ir as ir

from lib.symbol_table import SymbolTable
from lib.traversal import get_statement_lists
from lib.type_checks import TypeHelper
from lib.utils import unpack_iterated


def find_alias_groups(node: Union[ir.Function, ir.ForLoop, ir.WhileLoop], symbols: SymbolTable):
    graph = nx.Graph()
    typer = TypeHelper(symbols)
    for stmts in get_statement_lists(node):
        for stmt in stmts:
            if isinstance(stmt, ir.Assign):
                if typer.is_array(stmt.value):
                    target_name = get_target_name(stmt.target)
                    value_name = get_target_name(stmt.value)  # blank if this is out of place arithmetic
                    if target_name is not None and value_name is not None:
                        # avoids (a + b)[i] and other weird cases
                        graph.add_edge(target_name, value_name)
            elif isinstance(stmt, ir.ForLoop):
                for target, value in unpack_iterated(stmt.target, stmt.iterable):
                    if typer.iteration_yields_array(stmt.iterable) and isinstance(target, ir.NameRef):
                        alias_target = get_target_name(value)
                        if alias_target is not None:
                            graph.add_edge(target, alias_target)
    return nx.connected_components(graph)


def get_target_name(node: ir.ValueRef):
    if isinstance(node, ir.NameRef):
        return node
    elif isinstance(node, ir.Subscript):
        while isinstance(node, ir.Subscript):
            node = node.value
        if isinstance(node, ir.NameRef):
            return node


def find_non_aliasing_array_assigns(node: Union[ir.Function, ir.ForLoop, ir.WhileLoop], symbols: SymbolTable):
    nonaliasing = defaultdict(set)
    typer = TypeHelper(symbols)
    for stmts in get_statement_lists(node):
        for stmt in stmts:
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.target, ir.NameRef):
                    if typer.is_array(stmt.value):
                        if isinstance(stmt.value, ir.Expression) and not isinstance(stmt.value, ir.Subscript):
                            nonaliasing[stmt.target].add(stmt)
    return nonaliasing
