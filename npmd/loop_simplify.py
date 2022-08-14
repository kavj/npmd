import itertools
import numpy

import networkx as nx

from collections import defaultdict
from typing import Optional, Set

import npmd.ir as ir

from npmd.analysis import compute_element_count
from npmd.blocks import FlowGraph, build_function_graph, get_loop_exit_block, get_blocks_in_loop
from npmd.errors import CompilerError
from npmd.folding import fold_constants, simplify_arithmetic
from npmd.liveness import get_clobbers, get_loop_header_block, find_loop_iterables_clobbered_by_body, find_live_in_out
from npmd.lvn import ExprRewriter
from npmd.pretty_printing import PrettyFormatter
from npmd.reductions import NormalizeMinMax
from npmd.symbol_table import SymbolTable
from npmd.type_checks import invalid_loop_iterables, TypeHelper
from npmd.utils import unpack_iterated


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
        repl.extend(itertools.islice(header_stmt_list, header.stop, None))
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
    loops = [block.first for block in nx.descendants(graph.graph, graph.entry_block) if isinstance(block.first, ir.ForLoop)]
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


def make_affine_seq(node: ir.ValueRef):
    if isinstance(node, ir.AffineSeq):
        return node
    elif isinstance(node, ir.Subscript):
        index = node.index
        value = node.value
        if isinstance(index, ir.Slice):
            start = index.start
            step = index.step
            stop = ir.SingleDimRef(value, ir.Zero)
            if index.stop != ir.SingleDimRef(node.value, ir.Zero):
                stop = ir.MIN(index.stop, stop)
        else:
            start = ir.Zero
            stop = ir.SingleDimRef(value, ir.One)
            step = ir.One
        return ir.AffineSeq(start, stop, step)
    else:
        return ir.AffineSeq(ir.Zero, ir.SingleDimRef(node, ir.Zero), ir.One)


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

    typer = TypeHelper(symbols)
    normalize = NormalizeMinMax(symbols)

    by_iterable = {}
    target_to_iterable = {}
    for target, iterable in unpack_iterated(header.target, header.iterable):
        interval = make_affine_seq(iterable)
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
        affine_expr: ir.AffineSeq = exprs.pop()
    else:
        iterable_lens = []
        for expr in exprs:
            c = compute_element_count(expr.start, expr.stop, expr.step)
            iterable_lens.append(c)

        count = ir.MinReduction(*iterable_lens)
        affine_expr = ir.AffineSeq(ir.Zero, count, ir.One)

    loop_expr = normalize(affine_expr)
    assert isinstance(loop_expr, ir.AffineSeq)
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
        if isinstance(iterable, ir.AffineSeq):
            # we can reuse
            target_type = symbols.check_type(target)
            index_type = symbols.check_type(loop_index)
            if target_type == index_type:
                value = loop_index
            else:
                value = ir.CAST(loop_index, target_type)
        elif isinstance(iterable, ir.NameRef):
            value = ir.Subscript(iterable, loop_index)
        else:
            assert isinstance(iterable, ir.Subscript)
            if isinstance(iterable.index, ir.Slice):
                iterable_index = iterable.index
                if iterable_index.start == loop_expr.start and iterable_index.step == loop_expr.step:
                    value = ir.Subscript(iterable.value, loop_index)
                else:
                    assert loop_expr.step == ir.One and loop_expr.start == ir.Zero
                    if iterable_index.step != ir.One:
                        offset = ir.MULT(loop_index, iterable_index.step)
                        if iterable_index.start != ir.Zero:
                            offset = ir.ADD(offset, iterable_index.start)
                    elif iterable_index.start != ir.Zero:
                        offset = ir.ADD(loop_index, iterable_index.start)
                    else:
                        # slice doesn't augment start or step
                        offset = loop_index
                    value = ir.Subscript(iterable.value, offset)
            else:
                # single index, which means it's double sliced
                t = typer(iterable)
                name = symbols.make_unique_name_like(iterable, t)
                stmt = ir.Assign(name, iterable, pos)
                body.append(stmt)
                assert loop_expr.start == ir.Zero and loop_expr.step == ir.One
                value = ir.Subscript(name, loop_index)
        assign = ir.Assign(target, value, pos)
        body.append(assign)
    body.extend(header.body)
    header = ir.ForLoop(loop_index, loop_expr, body, pos)

    return header


def get_safe_loop_indices(stmt: ir.ForLoop, graph: FlowGraph, live_on_exit: Set[ir.NameRef]):
    header = get_loop_header_block(graph, stmt)
    blocks = get_blocks_in_loop(graph, header)
    cannot_use = live_on_exit.union(get_clobbers(blocks))
    safe_targets = []
    for target, iterable in unpack_iterated(stmt.target, stmt.iterable):
        if target not in cannot_use:
            safe_targets.append(target)
    return safe_targets


def lower_loops(graph: FlowGraph, symbols: SymbolTable):
    liveness = find_live_in_out(graph)
    loop_blocks = [block for block in graph.nodes() if block.is_loop_block and isinstance(block.first, ir.ForLoop)]
    headers = []
    # our graph is a volatile view rather than a transformable CFG, as this sidesteps issues of maintaining
    # full structure
    safe_target_lists = []
    for block in loop_blocks:
        # grab loop setup statement and any variables that are safe to use as loop indices
        header = block.first
        invalid_iterables = [*invalid_loop_iterables(header, symbols)]
        if invalid_iterables:
            formatter = PrettyFormatter()
            formatted = ', '.join(formatter(iterable) for iterable in invalid_iterables)
            msg = f'Non iterable references: {formatted} in for loop, line: {header.pos.line_begin}.'
            raise CompilerError(msg)
        headers.append(header)
        exit_block = get_loop_exit_block(graph, block)
        live_on_exit = liveness[exit_block].live_in
        safe_targets = get_safe_loop_indices(header, graph, live_on_exit)
        safe_target_lists.append(safe_targets)

    for header, no_escape in zip(headers, safe_target_lists):
        repl = make_single_index_loop(header, symbols, no_escape)
        header.target = repl.target
        header.iterable = repl.iterable
        header.body.clear()
        header.body.extend(repl.body)
