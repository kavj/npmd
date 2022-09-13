import itertools
import math
import numpy

from collections import defaultdict
from functools import singledispatch
from typing import Dict, List, Set, Tuple, Union

import npmd.ir as ir

from npmd.analysis import compute_element_count, extract_expressions
from npmd.blocks import build_function_graph, get_loop_exit_block
from npmd.errors import CompilerError
from npmd.folding import simplify
from npmd.liveness import find_live_in_out
from npmd.lvn import rewrite_expr
from npmd.pretty_printing import PrettyFormatter
from npmd.symbol_table import SymbolTable
from npmd.traversal import get_statement_lists, walk, walk_parameters
from npmd.type_checks import is_integer, TypeHelper
from npmd.utils import unpack_iterated


def invalid_loop_iterables(node: ir.ForLoop, symbols: SymbolTable):
    type_checker = TypeHelper(symbols)
    for _, iterable in unpack_iterated(node.target, node.iterable):
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


def serialize_min_max(node: Union[ir.MinReduction, ir.MaxReduction]):
    """
    Min max serialize without scraping all pairs
    :param node:
    :return:
    """
    if isinstance(node, ir.MinReduction):
        reducer = ir.MIN
    elif isinstance(node, ir.MaxReduction):
        reducer = ir.MAX
    else:
        msg = f"serializer requires min or max reduction. Received {node}."
        raise TypeError(msg)

    terms = list(node.subexprs)

    # serialize any nested terms
    for index, term in enumerate(terms):
        if isinstance(term, (ir.MaxReduction, ir.MinReduction)):
            terms[index] = serialize_min_max(term)

    num_terms = len(terms)
    if num_terms == 1:
        value = terms[0]
        return value
    if num_terms % 2:
        tail = terms[-1]
        terms = terms[:-1]
    else:
        tail = None
    step_count = math.floor(math.log2(len(terms)))

    for i in range(step_count):
        terms = [reducer(left, right) for left, right
                 in zip(itertools.islice(terms, 0, None, 2), itertools.islice(terms, 1, None, 2))]
    assert len(terms) == 1
    reduced = terms[0]
    if tail is not None:
        reduced = reducer(reduced, tail)
    return reduced


def add_trivial_return(node: ir.Function):
    """
    Method to add a trivial return to function.
    This will typically get cleaned up later if it's not reachable.
    :param node:
    :return:
    """

    if len(node.body) > 0:
        if isinstance(node.body[-1], ir.Return):
            return
        last = node.body[-1]
        pos = ir.Position(last.pos.line_end + 1, last.pos.line_end + 1, 1, 40)
        node.body.append(ir.Return(ir.NoneRef(), pos))
    else:
        pos = ir.Position(-1, -1, 1, 40)
        node.body.append(ir.Return(ir.NoneRef(), pos))


def preheader_rename_parameters(node: ir.ForLoop):
    """

    :param node:
    :return:
    """

    # Todo: we'll need a check for array augmentation as well later to see whether it's easily provable
    # that writes match the read pattern at entry

    clobbered_in_body = set()
    augmented_in_body = set()
    for stmts in get_statement_lists(node):
        for stmt in stmts:
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.target, ir.NameRef):
                    clobbered_in_body.add(stmt.target)
                else:
                    assert isinstance(stmt.target, ir.Subscript)
                    augmented_in_body.add(stmt.target.value)

            elif isinstance(stmt, ir.InPlaceOp):
                augmented_in_body.add(stmt.target)

            elif isinstance(stmt, ir.ForLoop):
                for target, _ in unpack_iterated(stmt.target, stmt.iterable):
                    assert isinstance(target, ir.NameRef)
                    clobbered_in_body.add(target)
    must_rename = set()

    for target, iterable in unpack_iterated(node.target, node.iterable):
        if isinstance(iterable, ir.AffineSeq):
            for param in walk_parameters(iterable):
                if param in clobbered_in_body or param in augmented_in_body:
                    must_rename.add(param)
        else:
            for param in walk_parameters(iterable):
                if param in clobbered_in_body:
                    must_rename.add(param)
    return must_rename


def rename_clobbered_loop_parameters(func: ir.Function, symbols: SymbolTable, verbose: bool = True):
    """
    This makes a copy of cases where the name used by the loop header is overwritten in the body,
    which makes analyzing their liveness much less confusing, particularly in the absence of LCSSA.

    :param func:
    :param symbols:
    :param verbose:
    :return:
    """
    # Todo: do we actually want this as a CFG dependent pass?
    typer = TypeHelper(symbols)
    for stmts in get_statement_lists(func):
        if any(isinstance(stmt, ir.ForLoop) for stmt in stmts):
            repl = []
            for stmt in stmts:
                if isinstance(stmt, ir.ForLoop):
                    clobbered = preheader_rename_parameters(stmt)
                    if clobbered:
                        renamed = {}
                        if verbose:
                            clobber_str = ', '.join(c.name for c in clobbered if isinstance(c, ir.NameRef))
                            msg = f'The following names appear in the list of loop iterables and are completely ' \
                                  f'overwritten in the loop body: "{clobber_str}"'
                            print(msg)
                        for c in clobbered:
                            if isinstance(c, ir.NameRef):
                                renamed[c] = symbols.make_versioned(c)
                            else:
                                t = typer(c)
                                renamed[c] = symbols.make_unique_name_like('v', t)
                        stmt.iterable = rewrite_expr(renamed, stmt.iterable)
                        pos = stmt.pos
                        for original, rename in renamed.items():
                            copy_stmt = ir.Assign(target=rename, value=original, pos=pos)
                            repl.append(copy_stmt)
                repl.append(stmt)
            stmts.clear()
            stmts.extend(repl)


def make_min_affine_seq(step, start_and_stops, symbols: SymbolTable):
    typer = TypeHelper(symbols)
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
            d = simplify(d, typer)
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
                if isinstance(stmt.target, ir.NameRef):

    """

    typer = TypeHelper(symbols)

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
        exprs.append(make_min_affine_seq(step, start_and_stop, symbols))

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

    loop_expr = simplify(affine_expr, typer)
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
        if target == loop_index:
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


def get_assigned_or_augmented(node: Union[ir.Function, ir.ForLoop, ir.WhileLoop]) -> Tuple[Set[ir.NameRef], Set[ir.NameRef]]:
    bound = set()
    augmented = set()
    for stmts in get_statement_lists(node):
        for stmt in stmts:
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.target, ir.NameRef):
                    bound.add(stmt.target)
                elif isinstance(stmt.target, ir.Subscript):
                    augmented.add(stmt.target.value)
            elif isinstance(stmt, ir.InPlaceOp):
                if isinstance(stmt.target, ir.NameRef):
                    augmented.add(stmt.target)
                else:
                    assert isinstance(stmt.target, ir.Subscript)
                    if isinstance(stmt.target.value, ir.NameRef):
                        augmented.add(stmt.target.value)
            elif isinstance(stmt, ir.ForLoop):
                # nested loop should not clobber
                for target, _ in unpack_iterated(stmt.target, stmt.iterable):
                    if isinstance(target, ir.NameRef):
                        bound.add(target)
    return bound, augmented


def get_safe_loop_indices(stmt: ir.ForLoop, live_on_exit: Set[ir.NameRef], symbols: SymbolTable):
    assigned, augmented = get_assigned_or_augmented(stmt)
    # It's technically safe for arrays to be augmented here, but we need to freeze any value
    # used to determine indexing
    assigned.update(augmented)
    assigned.update(live_on_exit)
    possible_indices = [target for (target, iterable) in unpack_iterated(stmt.target, stmt.iterable) if isinstance(iterable, ir.AffineSeq) and target not in assigned]
    # check that on the off chance of multiple assignment, we go by the last assigned
    invalid = set()
    for target, iterable in unpack_iterated(stmt.target, stmt.iterable):
        if not isinstance(iterable, ir.AffineSeq):
            invalid.add(target)
        else:
            invalid.discard(target)
    # Now extract any that are unambiguously typed as integers
    safe_targets = [target for target in possible_indices if is_integer(symbols.check_type(target))]
    return safe_targets


def lower_loops(func: ir.Function, symbols: SymbolTable, verbose: bool = False):
    rename_clobbered_loop_parameters(func, symbols, verbose)
    graph = build_function_graph(func)
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
        safe_targets = get_safe_loop_indices(header, live_on_exit, symbols)
        safe_target_lists.append(safe_targets)

    for header, no_escape in zip(headers, safe_target_lists):
        repl = make_single_index_loop(header, symbols, no_escape)
        header.target = repl.target
        header.iterable = repl.iterable
        header.body.clear()
        header.body.extend(repl.body)


def expand_in_place_assignments(entry_point: Union[ir.Function, ir.ForLoop, ir.WhileLoop], typer: TypeHelper):
    for stmts in get_statement_lists(entry_point):
        for index, stmt in enumerate(stmts):
            if isinstance(stmt, ir.InPlaceOp):
                if not isinstance(typer(stmt.target), ir.ArrayType):
                    stmts[index] = ir.Assign(stmt.target, stmt.value, stmt.pos)


def get_array_inits(stmt: ir.StmtBase):
    for expr in extract_expressions(stmt):
        for subexpr in walk(expr):
            if isinstance(subexpr, ir.ArrayInitializer):
                yield subexpr


def make_array_init_assigns(inits: List[ir.ArrayInitializer], symbols: SymbolTable, pos: ir.Position):
    typer = TypeHelper(symbols)
    by_param = defaultdict(list)
    stmts = []
    # repl.append(ir.ArrayFill(target, stmt.value.fill_value, pos))
    for init in inits:
        t = typer(init)
        name = symbols.make_unique_name_like('a', t)
        alloc = ir.ArrayAlloc(init.shape, init.dtype)
        assign = ir.Assign(name, alloc, pos)
        stmts.append(assign)
        by_param[init].append(name)
        if not isinstance(init.fill_value, ir.NoneRef):
            stmt = ir.ArrayFill(name, init.fill_value, pos)
            stmts.append(stmt)
    return stmts, by_param


def substitute_array_refs(node: ir.ValueRef, array_refs: Dict[ir.ArrayInitializer, List[ir.NameRef]]):
    if isinstance(node, ir.Expression):
        if isinstance(node, ir.ArrayInitializer):
            repl = array_refs[node].pop()
            if not array_refs[node]:
                array_refs.pop(node)
            return repl
        else:
            return node.reconstruct(*(substitute_array_refs(subexpr, array_refs) for subexpr in node.subexprs))
    else:
        return node


@singledispatch
def replace_initializers(node, array_refs: Dict[ir.ArrayInitializer, List[ir.NameRef]]):
    msg = f'No method to replace initializers for node {node}'
    raise TypeError(msg)


@replace_initializers.register
def _(node: ir.StmtBase, array_refs: Dict[ir.ArrayInitializer, List[ir.NameRef]]):
    return node


@replace_initializers.register
def _(node: ir.Assign, array_refs: Dict[ir.ArrayInitializer, List[ir.NameRef]]):
    if any(isinstance(subexpr, ir.ArrayInitializer) for subexpr in walk(node.target)):
        # this would have to be something like numpy.zeros(n)[i] = 42
        msg = f'Array initialization functions cannot be part of an assignment target: line {node.pos}'
        raise CompilerError(msg)
    value = substitute_array_refs(node.value, array_refs)
    if value == node.value:
        return node
    else:
        return ir.Assign(node.target, value, node.pos)


@replace_initializers.register
def _(node: ir.InPlaceOp, array_refs: Dict[ir.ArrayInitializer, List[ir.NameRef]]):
    if any(isinstance(subexpr, ir.ArrayInitializer) for subexpr in walk(node.target)):
        # this would have to be something like numpy.zeros(n)[i] = 42
        msg = f'Array initialization functions cannot be part of an assignment target: line {node.pos}'
        raise CompilerError(msg)
    value = substitute_array_refs(node.value, array_refs)
    if value == node.value:
        return node
    else:
        value = substitute_array_refs(node.value, array_refs)
        return ir.InPlaceOp(node.target, value, node.pos)


@replace_initializers.register
def _(node: ir.Return, array_refs: Dict[ir.ArrayInitializer, List[ir.NameRef]]):
    value = substitute_array_refs(node.value, array_refs)
    return ir.Return(value, node.pos)


@replace_initializers.register
def _(node: ir.SingleExpr, array_refs: Dict[ir.ArrayInitializer, List[ir.NameRef]]):
    value = substitute_array_refs(node.value, array_refs)
    if value == node.value:
        return node
    else:
        return ir.SingleExpr(value, node.pos)


@replace_initializers.register
def _(node: ir.IfElse, array_refs: Dict[ir.ArrayInitializer, List[ir.NameRef]]):
    test = substitute_array_refs(node.test, array_refs)
    if test != node.test:
        node.test = test
    return node


@replace_initializers.register
def _(node: ir.WhileLoop, array_refs: Dict[ir.ArrayInitializer, List[ir.NameRef]]):
    test = substitute_array_refs(node.test, array_refs)
    if test != node.test:
        node.test = test
    return node


def split_array_assignment(node: ir.Assign):
    """
    split initialization where we don't require a temporary vable
    :param node:
    :return:
    """
    assert isinstance(node.target, ir.NameRef)
    assert isinstance(node.value, ir. ArrayInitializer)
    value = node.value
    alloc = ir.ArrayAlloc(value.shape, value.dtype)
    assign = ir.Assign(node.target, alloc, node.pos)
    if isinstance(value.fill_value, ir.NoneRef):
        stmts = [assign]
    else:
        fill = ir.ArrayFill(node.target, value.fill_value, node.pos)
        stmts = [assign, fill]
    return stmts


def normalize_array_initializers(func: ir.Function, symbols: SymbolTable):
    """
    Splits initialization from allocation. This gives us some ability to hoist or remove large allocations.

    :param func:
    :param symbols:
    :return:
    """
    repl = []
    for stmts in get_statement_lists(func):
        has_init = False
        for stmt in stmts:
            array_inits = list(get_array_inits(stmt))
            if array_inits:
                has_init = True
                if isinstance(stmt, ir.Assign) and isinstance(stmt.target, ir.NameRef) and isinstance(stmt.value, ir.ArrayInitializer):
                    repl.extend(split_array_assignment(stmt))
                else:
                    repl_stmts, by_param = make_array_init_assigns(array_inits, symbols, stmt.pos)
                    repl.extend(repl_stmts)
                    repl_stmt = replace_initializers(stmt, by_param)
                    repl.append(repl_stmt)
            else:
                repl.append(stmt)

        if has_init:
            stmts.clear()
            stmts.extend(repl)
            repl.clear()
