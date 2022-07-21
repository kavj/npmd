import itertools

import numpy

from collections import defaultdict
from typing import Iterable, Optional

import ir

from analysis import compute_element_count
from blocks import FlowGraph, build_function_graph, get_loop_exit_block
from folding import simplify_arithmetic, fold_constants
from liveness import get_loop_header_block, find_loop_iterables_clobbered_by_body, find_live_in_out
from lvn import ExprRewriter
from symbol_table import SymbolTable
from traversal import walk_nodes
from utils import unpack_iterated


def rename_clobbered_loop_iterables(graph: FlowGraph, loop: ir.ForLoop, symbols: SymbolTable, verbose: bool = True):
    """
    This makes a copy of cases where the name used by the loop header is overwritten in the body,
    which makes analyzing their liveness much less confusing, particularly in the absence of LCSSA.

    :param graph:
    :param loop:
    :param symbols:
    :param verbose:
    :return:
    """
    # Todo: do we actually want this as a CFG dependent pass?
    header = get_loop_header_block(graph, loop)
    clobbered = find_loop_iterables_clobbered_by_body(graph, header)
    if clobbered:
        renamed = {}
        if verbose:
            clobber_str = ', '.join(c.name for c in clobbered)
            msg = f'The following names appear in the list of loop iterables and are completely ' \
                  f'overwritten in the loop body: "{clobber_str}"'
            print(msg)
        for c in clobbered:
            renamed[c] = symbols.make_versioned(c)
        rewriter = ExprRewriter(symbols, renamed)
        iterable_expr = rewriter.rewrite_expr(loop.iterable)
        loop.iterable = iterable_expr
        pos = loop.pos
        header_stmt_list = header.statements
        start = header.start
        # reinsert everything up to the loop
        repl = header_stmt_list[:start]
        for original, rename in renamed.items():
            stmt = ir.Assign(target=rename, value=original, pos=pos)
            repl.append(stmt)
        repl.append(loop)
        # add remainder
        repl.extend(itertools.islice(header_stmt_list, header.stop))
        # now replace the original statements the first list
        header_stmt_list.clear()
        header_stmt_list.extend(repl)


def sanitize_loop_iterables(node: ir.Function,
                            symbols: SymbolTable,
                            graph: Optional[FlowGraph] = None,
                            verbose: bool = True):
    """
    This shallow copies any iterables clobbered by a loop body, prior to loop body entry and substitutes
    the copy name in the header references.
    :param node:
    :param symbols:
    :param graph:
    :param verbose:
    :return:
    """
    if graph is None:
        graph = build_function_graph(node)
    loops = [stmt for stmt in walk_nodes(node.body) if isinstance(stmt, ir.ForLoop)]
    for loop in loops:
        rename_clobbered_loop_iterables(graph, loop, symbols, verbose)


def make_min_affine_seq(step, start_and_stops):
    starts = set()
    stops = set()
    for start, stop in start_and_stops:
        starts.add(start)
        stops.add(stop)
    stops.discard(None)
    if len(starts) == 1:
        start = starts.pop()
        if len(stops) == 1:
            stop = stops.pop()
        else:
            stop = ir.MinReduction(*stops)
        return ir.AffineSeq(start, stop, step)
    elif len(stops) == 1:
        stop = stops.pop()
        stop = ir.MAX(ir.SUB(stop, ir.MaxReduction(*starts)), ir.Zero)
        # need a zero here, since
        return ir.AffineSeq(ir.Zero, stop, step)
    else:
        diffs = []
        for start, stop in start_and_stops:
            d = ir.SUB(stop, start)
            d = simplify_arithmetic(d)
            d = fold_constants(d)
            diffs.append(d)
        min_diff = ir.MAX(ir.MinReduction(*diffs), ir.Zero)
        return ir.AffineSeq(ir.Zero, min_diff, step)


def affine_sequence_from_iterable(node: ir.ValueRef):
    if isinstance(node, ir.AffineSeq):
        return node
    elif isinstance(node, ir.NameRef):
        return ir.AffineSeq(ir.Zero, ir.SingleDimRef(node, ir.Zero), ir.One)
    elif isinstance(node, ir.Subscript):
        index = node.index
        value = node.value
        if isinstance(index, ir.Slice):
            start = index.start
            step = index.step
            stop = ir.SingleDimRef(value, ir.Zero)
            if index.stop is not None:
                stop = ir.MIN(index.stop, stop)
        else:
            start = ir.Zero
            stop = ir.SingleDimRef(value, ir.One)
            step = ir.One
        return ir.AffineSeq(start, stop, step)
    else:
        msg = f'No method to make affine sequence from "{node}".'
        raise TypeError(msg)


def make_single_index_loop(header: ir.ForLoop, symbols, noescape):
    """

        Make loop interval of the form (start, stop, step).

        This tries to find a safe method of calculation.

        This assumes (with runtime verification if necessary)
        that 'stop - start' will not overflow.

        References:
            LIVINSKII et. al, Random Testing for C and C++ Compilers with YARPGen
            Dietz et. al, Understanding Integer Overflow in C/C++
            Bachmann et. al, Chains of Recurrences - a method to expedite the evaluation of closed-form functions
            https://developercommunity.visualstudio.com/t/please-implement-integer-overflow-detection/409051
            https://numpy.org/doc/stable/user/building.html


        Note: this can complicate range analysis if it forces normalization, so it should be performed late
        # Todo: add reference, think it's a Michael Wolf paper that mentions this

    """

    by_iterable = {}
    target_to_iterable = {}
    for target, iterable in unpack_iterated(header.target, header.iterable):
        interval = affine_sequence_from_iterable(iterable)
        by_iterable[iterable] = interval
        target_to_iterable[target] = iterable

    by_step = defaultdict(set)
    for interval in by_iterable.values():
        by_step[interval.step].add((interval.start, interval.stop))

    exprs = []
    for step, start_and_stop in by_step.items():
        exprs.append(make_min_affine_seq(step, start_and_stop))

    if len(exprs) == 1:
        # we can avoid normalizing
        loop_expr = exprs.pop()
    else:
        iterable_lens = []
        for loop_expr in exprs:
            c = compute_element_count(loop_expr.start, loop_expr.stop, loop_expr.step)
            iterable_lens.append(c)
        count = ir.MinReduction(*iterable_lens)
        loop_expr = ir.AffineSeq(ir.Zero, count, ir.One)

    loop_index = None
    for target in target_to_iterable:
        iterable = target_to_iterable[target]
        if loop_index is None:
            if isinstance(iterable, ir.AffineSeq):
                if iterable.start == loop_expr.start and iterable.step == loop_expr.step and target in noescape:
                    loop_index = target
    # we only use an existing variable as an index if the variable name doesn't escape the loop
    # since the index itself is assigned when testing whether to enter the loop rather than right before entry
    if loop_index is None:
        loop_index = symbols.make_unique_name_like('i', numpy.dtype('int64'))

    body = []
    pos = header.pos
    for target, iterable in unpack_iterated(header.target, header.iterable):
        if target is loop_index:
            continue
        seq = by_iterable[iterable]
        if seq.start == loop_expr.start:
            if seq.step == loop_expr.step:
                value = loop_index
            else:
                assert loop_expr.start == ir.Zero and loop_expr.step == ir.One
                value = ir.MULT(iterable.step, loop_index)
                value = fold_constants(value)
        elif iterable.step == loop_expr.step:
            assert loop_expr.start == ir.Zero
            value = ir.ADD(iterable.start, loop_index)
        else:
            assert loop_expr.start == ir.Zero and loop_expr.step == ir.One
            offset = ir.MULT(iterable.step, loop_index)
            value = ir.ADD(iterable.start, offset)
        assign = ir.Assign(target, value, pos)
        body.append(assign)
    body.extend(header.body)
    header = ir.ForLoop(loop_index, loop_expr, body, pos)

    return header


def lower_loops(stmts: Iterable[ir.Statement], symbols: SymbolTable, graph: FlowGraph):
    liveness = find_live_in_out(graph)
    headers = [node for node in walk_nodes(stmts) if isinstance(node, ir.ForLoop)]
    # graph is invalidated by changes, so we have to get all nodes in advance
    no_escapes = []
    for header in headers:
        exit_block = get_loop_exit_block(graph, header)
        live_on_exit = liveness[exit_block].live_in
        no_escape_targets = {t for (t, i) in unpack_iterated(header.target, header.iterable) if t not in live_on_exit}
        no_escapes.append(no_escape_targets)

    for header, no_escape in zip(headers, no_escapes):
        repl = make_single_index_loop(header, symbols, no_escape)
        header.target = repl.target
        header.iterable = repl.iterable
        header.body.clear()
        header.body.extend(repl.body)
