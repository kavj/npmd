
import networkx as nx

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Set

import npmd.ir as ir

from npmd.analysis import extract_expressions, extract_parameters
from npmd.blocks import BasicBlock
from npmd.liveness import BlockLiveness
from npmd.lvn import rename_dead_on_exit
from npmd.symbol_table import SymbolTable
from npmd.traversal import walk
from npmd.type_checks import TypeHelper


def get_block_entry_array_constraints(block: BasicBlock, liveness: BlockLiveness, typer: TypeHelper):
    """
    This finds array constraints that must hold before any clobbers are made.

    Specifically, even if augmented, applying a op b where a and b are both arrays
    is an error if these do not have compatible dimensions, and we want to determine
    if any of these assumptions are violated prior to block entry.

    :param liveness:
    :param block:
    :param typer:
    :return:
    """
    if block.is_entry_block or block.is_entry_point:
        return
    # check for ephemeral
    counts = defaultdict(int)
    clobbers = set()
    # we don't necessarily encounter these in order
    # thus the need for a graph
    graph = nx.Graph()

    for stmt in block:
        if isinstance(stmt, ir.Assign) and isinstance(stmt.target, ir.NameRef):
            counts[stmt.target] += 1

    ephemeral = {name for (name, count) in counts.items() if name not in liveness.live_in and count == 1}

    for stmt in block:
        # get expressions
        for expr in extract_expressions(stmt):
            for subexpr in walk(expr):
                # wondering if this should restrict to all unclobbered..
                if isinstance(subexpr, ir.Expression):
                    if typer.is_array(subexpr):
                        # if subexpr is an array at this level, then this constrains
                        # the dimensions of any directly nested array sub-expression
                        for s in subexpr.subexprs:
                            if typer.is_array(s):
                                if isinstance(s, ir.Expression):
                                    # need to know if these are safe
                                    # TODO: finish implementation
                                    if s in ephemeral or s not in clobbers:
                                        graph.add_edge(subexpr, s)

        if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.NameRef):
            # for this pass, don't consider binding operations that are not live-in, since
            # we may not be able to determine exact constraints at block entry.
            clobbers.add(stmt.target)

    return graph


@dataclass
class ParallelLoop:
    induction_target: ir.ValueRef
    induction_expr: ir.AffineSeq
    stmts: List[ir.StmtBase]


def fuse_regions(block: BasicBlock, liveness: BlockLiveness, symbols: SymbolTable):
    rename_dead_on_exit(block, liveness, symbols)


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
