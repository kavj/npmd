import itertools

import networkx as nx

from dataclasses import dataclass
from typing import Iterable, List, Set

import npmd.ir as ir

from npmd.analysis import extract_expressions, extract_parameters
from npmd.blocks import BasicBlock
from npmd.liveness import BlockLiveness
from npmd.symbol_table import SymbolTable
from npmd.traversal import walk
from npmd.type_checks import TypeHelper


def get_array_expressions(expr: ir.ValueRef, typer: TypeHelper):
    """
    Get expressions that are part of the current expression, which have an array valued output.
    :param expr:
    :param typer:
    :return:
    """
    for subexpr in walk(expr):
        if isinstance(subexpr, ir.Expression) and typer.is_array(subexpr):
            yield subexpr


def seed_array_expr_constraints(block: BasicBlock, liveness: BlockLiveness, typer: TypeHelper):
    """
    This finds the initial array constraints. This avoids the problem
    :param block:
    :param liveness:
    :param typer:
    :return:
    """
    if block.is_entry_block or block.is_loop_block or block.is_branch_block:
        return
    relevant = set(liveness.live_in)
    for stmt in block:
        if isinstance(stmt, (ir.Assign, ir.InPlaceOp)):
            pass
    pass


class ArrayLengthTracking:

    def __init__(self, symbols: SymbolTable):
        self.graph = nx.Graph()
        self.typer = TypeHelper(symbols)

    def get_tracked(self):
        # useful to distinguish what's new
        for node in self.graph.nodes():
            yield node

    def add_expr(self, expr: ir.ValueRef):
        # add correlation bounds constraint
        for subexpr in walk(expr):
            if isinstance(subexpr, ir.Expression) and self.typer.is_array(subexpr):
                for dep in subexpr.subexprs:
                    if self.typer.is_array(dep):
                        # If this is an elementwise type (should be checked better), we can correlate
                        # the parent expression with any direct array dependency, so if a,b,c are 1d arrays,
                        # a + b * min(c) would correlate 'a' and 'b'.
                        # Note that if we clobber
                        self.graph.add_edge(subexpr, dep)

    def bind_expr(self, target: ir.ValueRef, value: ir.ValueRef):
        # make sure value is bound
        self.add_expr(value)
        if target == value:
            # don't make self edges.
            return
        elif not self.typer.is_array(target):
            return
        assert self.typer.is_array(value)
        if isinstance(target, ir.NameRef):
            if isinstance(value, ir.Expression) and target in value.subexprs:
                # no changes, as we have already added the right-hand side constraint
                return
            else:
                # We have to break initial constraints on this variable, since it's being rebound
                # There's probaby a better way to do this, but for now, preserve existing constraints
                # connecting the neighbors of target
                for u, v in itertools.product(nx.neighbors(self.graph, target)):
                    self.graph.add_edge(u, v)
                self.graph.remove_node(target)
        else:
            # if this is a sliced array, capture it
            self.add_expr(target)
        self.graph.add_edge(target, value)


def link_correlated_expressions(block: Iterable[ir.StmtBase], typer: TypeHelper)->nx.Graph:
    """
    This splits any contiguous sequence of statements into fusable ones.
    :param block:
    :param typer:
    :return:
    """

    clusters = nx.Graph()
    for stmt in block:
        for exprs in extract_expressions(stmt):
            for expr in exprs:
                for array_expr in get_array_expressions(expr, typer):
                    # gather directly nested sub-expressions, which are also arrays
                    array_subexprs = [subexpr for subexpr in array_expr.subexprs if typer.is_array(subexpr)]
                    clusters.add_node(array_expr)
                    for next_index, subexpr in enumerate(array_subexprs, 1):
                        for other in itertools.islice(array_subexprs, next_index, None):
                            clusters.add_edge(array_expr, other)
    return clusters


@dataclass
class ParallelLoop:
    induction_target: ir.ValueRef
    induction_expr: ir.AffineSeq
    stmts: List[ir.StmtBase]


def get_array_statements(stmts: Iterable[ir.StmtBase], typer: TypeHelper):
    for stmt in stmts:
        for parameter in extract_parameters(stmt):
            if typer.is_array(parameter):
                yield stmt


def is_array_stmt(stmt: ir.StmtBase, typer: TypeHelper):
    return any((typer.is_array(parameter) for parameter in expr) for expr in extract_parameters(stmt))


def partition_block(stmts: BasicBlock, symbols: SymbolTable):
    typer = TypeHelper(symbols)
    partitions = []
    current = None
    is_array_partition = None
    for stmt in stmts:
        if current:
            if is_array_partition == is_array_stmt(stmt, typer):
                current.append(stmt)
                continue
            else:
                partitions.append(current)
        is_array_partition = is_array_stmt(stmt, typer)
        current = [stmt]
    if current is not None:
        partitions.append(current)
    # Now we need to determine what can actually fuse
    # The legality criteria comes down to shared iteration bounds
    # The secondary criteria is whether uses are in fact correlated.
    # If not, it's a bad idea to fuse, as it convolutes access patterns
    # and increases register pressure.

    return partitions


def make_parallel_loop(stmts: BasicBlock,
                       alias_groups: List[Set[ir.NameRef]],
                       liveness: BlockLiveness,
                       symbols: SymbolTable):
    typer = TypeHelper(symbols)
    partitions = partition_block(stmts, symbols)
    for partition in partitions:
        if is_array_stmt(partition[0], typer):
            # Now we need to find the fusable sub-regions here. This might be interrupted by scalar expression
            # but for now, we aren't reorganizing those.

            pass
    for stmt in stmts:
        if is_array_stmt(stmt, typer):

            pass

    # verify we have array statements

    # check which may be fusable

    # fusion ends if we need to carry a different access pattern
    # or may clobber

    pass

