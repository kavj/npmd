import numpy

import networkx as nx
import lib.ir as ir

from collections import defaultdict
from functools import singledispatch
from typing import Set

from lib.blocks import BasicBlock, FunctionContext, make_temporary_assign
from lib.errors import CompilerError
from lib.expression_utils import find_element_count, make_affine_seq, make_min_affine_seq, remap_parameters
from lib.folding import simplify_untyped_numeric
from lib.formatting import PrettyFormatter
from lib.walkers import get_blocks_in_loop, get_reduced_graph, get_loop_entry_block, get_loop_exit_block, \
    insert_block_before, walk_graph
from lib.liveness import find_live_in_out
from lib.statement_utils import get_assigned_or_augmented
from lib.symbol_table import SymbolTable
from lib.type_checks import is_integer, TypeHelper
from lib.unpacking import unpack_loop_iter, unpack_iter


@singledispatch
def get_iterator_access_func(node):
    raise NotImplementedError


@get_iterator_access_func.register
def _(node: ir.NameRef):
    return ir.Zero, ir.SingleDimRef(node, ir.Zero), ir.One


@get_iterator_access_func.register
def _(node: ir.Subscript):
    if isinstance(node.index, ir.Slice):
        start = ir.Zero if node.index.start is None else node.index.start
        max_stop = ir.SingleDimRef(node, ir.Zero)
        stop = max_stop if node.index.stop is None else ir.MIN(max_stop, node.index.stop)
        step = ir.One if node.index.step is None else node.index.step
        return start, stop, step
    else:
        return ir.Zero, ir.SingleDimRef(node, ir.One), ir.One


def get_access_ranges(header: ir.ForLoop):
    for _, iterable in unpack_loop_iter(header):
        yield get_iterator_access_func(iterable)


def invalid_loop_iterables(node: ir.ForLoop, symbols: SymbolTable):
    type_checker = TypeHelper(symbols)
    for _, iterable in unpack_loop_iter(node):
        if isinstance(iterable, ir.AffineSeq):
            for subexpr in iterable.subexprs:
                # Todo: raise here instead?
                if not is_integer(type_checker(subexpr)):
                    yield subexpr
        elif isinstance(iterable, ir.Subscript):
            if isinstance(iterable.index, ir.Slice):
                for subexpr in iterable.index.subexprs:
                    if not is_integer(type_checker(subexpr)):
                        yield subexpr
            elif not is_integer(type_checker(iterable.index)):
                yield iterable.index
        else:
            t = type_checker(iterable)
            if not isinstance(t, ir.ArrayType):
                yield iterable


def rename_clobbered_loop_parameters(func: FunctionContext,  block: BasicBlock):
    """
    This makes a copy of cases where the name used by the loop header is overwritten in the body,
    which makes analyzing their liveness much less confusing, particularly in the absence of LCSSA.

    :param func:
    :param block:
    :return:
    """

    assert block.is_loop_block
    liveness = find_live_in_out(func.graph)
    # Since we're concerned about things crossing the loop header, we should look at SCCs for this loop and ignore
    # divergent components
    blocks = get_blocks_in_loop(func.graph, block)
    targets = {target for (target, _) in unpack_loop_iter(block.first) if isinstance(target, ir.NameRef)}

    reassigned = set()
    for b in blocks:
        block_liveness = liveness[b]
        reassigned.update(block_liveness.kills)

    must_rename = targets.intersection(reassigned)

    if must_rename:
        reduced = get_reduced_graph(func)
        pos = block.first.pos
        assigns = [make_temporary_assign(func, name, name, pos) for name in must_rename]
        old_to_new = {stmt.value: stmt.target for stmt in assigns}
        # after pruning back edges, we should only have
        if reduced.in_degree[block] != 1:
            # not well-formed, possibly unreachable latch or loop body
            raise CompilerError()
        parent, = reduced.predecessors(block)
        # if we have a non-entry point block preceding the loop header, we can append to that, otherwise
        # this requires an explicit pre-header
        if parent.is_entry_point:
            insert_block_before(func, block, assigns)
        else:
            for s in assigns:
                parent.append_statement(s)
        header = block.first
        target = remap_parameters(header.iterable, old_to_new)
        iterable = remap_parameters(header.target, old_to_new)
        # Now we need to substitute the renamed ones into the loop
        repl_header = ir.ForLoop(target, iterable, header.pos)
        block.replace_statement(repl_header, 0)


def preheader_rename_parameters(func: FunctionContext):
    """
    :param func:
    :return:


    Renames anything with body clobbers, including those where the clobber happens in a divergent block.

    """

    outer_blocks = [b for b in nx.descendants(func.graph, func.entry_point) if b.is_loop_block and b.depth == 0]
    expanded = set()
    for header in outer_blocks:
        expanded.add(header)
        queued = [b for b in get_blocks_in_loop(func.graph, header) if b.is_loop_block]
        while queued:
            next_header = queued[-1]
            if next_header in expanded:
                queued.pop()
                if isinstance(next_header.first, ir.ForLoop):
                    rename_clobbered_loop_parameters(func, next_header)
            else:
                queued.extend(b for b in get_blocks_in_loop(func.graph, next_header) if b.is_loop_block)
                expanded.add(next_header)


def make_single_index_loop(func: FunctionContext, header_block: BasicBlock, noescape: Set[ir.NameRef]):
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
                if isinstance(stmt.target, ir.NameRef):

    """

    assert header_block.is_loop_block and isinstance(header_block.first, ir.ForLoop)
    typer = TypeHelper(func.symbols)

    by_iterable = {}
    target_to_iterable = {}
    for target, iterable in unpack_loop_iter(header_block.first):
        interval = make_affine_seq(iterable)
        by_iterable[iterable] = interval
        target_to_iterable[target] = iterable

    by_step = defaultdict(set)
    for interval in by_iterable.values():
        by_step[interval.step].add((interval.start, interval.stop))
    exprs = []
    for step, start_and_stop in by_step.items():
        exprs.append(make_min_affine_seq(step, start_and_stop, func.symbols))

    if len(exprs) == 1:
        # we can avoid normalizing
        affine_expr: ir.AffineSeq = exprs.pop()
    else:
        iterable_lens = []
        for expr in exprs:
            c = find_element_count(expr)
            iterable_lens.append(c)

        count = ir.MinOf(*iterable_lens)
        affine_expr = ir.AffineSeq(ir.Zero, count, ir.One)

    loop_expr = simplify_untyped_numeric(affine_expr)
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
        loop_index = func.symbols.make_unique_name_like('i', numpy.dtype('int64'))

    body = []
    header = header_block.first
    pos = header.pos
    for target, iterable in unpack_loop_iter(header):
        if target == loop_index:
            continue
        if isinstance(iterable, ir.AffineSeq):
            # we can reuse
            target_type = func.symbols.check_type(target)
            index_type = func.symbols.check_type(loop_index)
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
                name = func.symbols.make_unique_name_like(iterable, t)
                stmt = ir.Assign(name, iterable, pos)
                body.append(stmt)
                assert loop_expr.start == ir.Zero and loop_expr.step == ir.One
                value = ir.Subscript(name, loop_index)
        assign = ir.Assign(target, value, pos)
        body.append(assign)

    loop_body_entry_block = get_loop_entry_block(func, header_block)
    assert loop_body_entry_block is not None
    if loop_body_entry_block.is_entry_point:
        # we have a perfectly nested loop before unwinding the iterator
        # (probably want to do this as late as possible, as it breaks perfect nesting)
        # This means that we need to gather this setup in a new block, which dominates the nested entry
        func.graph.remove_edge(header, loop_body_entry_block)
        func.add_block(body, loop_body_entry_block.depth, [header_block], [loop_body_entry_block])
    else:
        body.extend(loop_body_entry_block)
        loop_body_entry_block.replace_statements(body)
        header = ir.ForLoop(loop_index, loop_expr, pos)
        header_block.replace_statement(header, 0)

    return header_block


def get_safe_loop_indices(func: FunctionContext, block: BasicBlock, live_on_exit: Set[ir.NameRef]):
    assert block.is_loop_block
    assigned, augmented = get_assigned_or_augmented(func, block)
    # It's technically safe for arrays to be augmented here, but we need to freeze any value
    # used to determine indexing
    assigned.update(augmented)
    assigned.update(live_on_exit)
    stmt = block.first
    possible_indices = [target for (target, iterable) in unpack_loop_iter(stmt) if isinstance(iterable, ir.AffineSeq)
                        and target not in assigned]
    # check that on the off chance of multiple assignment, we go by the last assigned
    invalid = set()
    for target, iterable in unpack_loop_iter(stmt):
        if not isinstance(iterable, ir.AffineSeq):
            invalid.add(target)
        else:
            invalid.discard(target)
    # Now extract any that are unambiguously typed as integers
    safe_targets = [target for target in possible_indices if is_integer(func.symbols.check_type(target))]
    return safe_targets


def lower_loops(func: FunctionContext):
    graph = func.graph
    symbols = func.symbols
    liveness = find_live_in_out(graph)
    loop_blocks = [block for block in walk_graph(func.graph) if block.is_loop_block and isinstance(block.first, ir.ForLoop)]
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
        exit_block = get_loop_exit_block(func, block)
        live_on_exit = liveness[exit_block].live_in
        safe_targets = get_safe_loop_indices(func, block, live_on_exit)

        safe_target_lists.append(safe_targets)
        rename_clobbered_loop_parameters(func, block)

    raise NotImplementedError('This needs a rewrite')
