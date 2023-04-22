from __future__ import annotations

import itertools
import math

from typing import Dict, Optional, Union

import lib.ir as ir

from lib.errors import CompilerError
from lib.folding import simplify_untyped_numeric
from lib.symbol_table import SymbolTable
from lib.walkers import walk_expr
from lib.type_checks import TypeHelper

# Todo: stub
specialized = {ir.NameRef("print")}


def is_constant(value: ir.ValueRef):
    if isinstance(value, ir.CONSTANT):
        return True
    elif isinstance(value, ir.Expression):
        for subexpr in walk_expr(value):
            # if we have any possibly non-constant expression, return False
            if isinstance(subexpr, (ir.Call, ir.NameRef)):
                return False


def remap_parameters(expr: ir.ValueRef, cached: Dict[ir.ValueRef, ir.ValueRef]):
    repl = cached.get(expr)
    if repl is None:
        if isinstance(expr, ir.Expression):
            subexprs = [remap_parameters(subexpr, cached) for subexpr in expr.subexprs]
            repl = expr.reconstruct(*subexprs)
            cached[expr] = repl
        else:
            repl = expr
    return repl


def serialize_min_max(node: Union[ir.MinOf, ir.MaxOf]):
    """
    Min max serialize without scraping all pairs
    :param node:
    :return:
    """
    if isinstance(node, ir.MinOf):
        reducer = ir.MIN
    elif isinstance(node, ir.MaxOf):
        reducer = ir.MAX
    else:
        msg = f"serializer requires min or max reduction. Received {node}."
        raise TypeError(msg)

    terms = list(node.subexprs)

    # serialize any nested terms
    for index, term in enumerate(terms):
        if isinstance(term, (ir.MaxOf, ir.MinOf)):
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
            stop = ir.MinOf(*stops)
        return ir.AffineSeq(start, stop, step)
    elif len(stops) == 1:
        stop = stops.pop()
        stop = ir.MAX(ir.SUB(stop, ir.MaxOf(*starts)), ir.Zero)
        # need a zero here, since
        return ir.AffineSeq(ir.Zero, stop, step)
    else:
        diffs = []
        for start, stop in start_and_stops:
            d = ir.SUB(stop, start)
            d = simplify_untyped_numeric(d)
            diffs.append(d)
        min_diff = ir.MAX(ir.MinOf(*diffs), ir.Zero)
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


def expression_strictly_contains(a: ir.ValueRef, b: ir.ValueRef):
    """
    returns True if evaluation of 'b' requires evaluation of 'a'.

    :param a:
    :param b:
    :return:
    """

    if not isinstance(b, ir.Expression) or a == b:
        return False
    for subexpr in walk_expr(b):
        if subexpr == a:
            return True
    return False


def greatest_common_subexprs(a: ir.ValueRef, b: ir.ValueRef):
    """
    returns the broadest sub-expressions that are common to 'a' and 'b'.
    :param a:
    :param b:
    :return:
    """

    if a == b:
        return a

    subexprs_a = set(walk_expr(a))
    subexprs_b = set(walk_expr(b))

    # get subexpression overlap if any
    common = subexprs_a.intersection(subexprs_b)
    if not common:
        return common

    sub = set()
    for u in common:
        for v in common:
            if expression_strictly_contains(v, u):
                # remove all expressions which are sub-expressions of any common sub-expression
                sub.add(u)
    common.difference_update(sub)
    return common


def find_element_count(node: ir.ValueRef) -> Optional[ir.ValueRef]:
    """
      Find an expression for array length if countable. Otherwise returns None.
      For example:
          a[a > 0] is not trivially countable.
          The predicate must be evaluated against the array
          to determine the number of elements.

          a[i::] is trivially countable. For i >= 0, len(a[i::]) == len(a) - i

      :return:
    """

    if isinstance(node, ir.Subscript):
        if isinstance(node.index, ir.Slice):
            index = node.index
            start = index.start
            stop = index.stop
            base_len = find_element_count(node.value)
            stop = ir.MIN(stop, base_len) if (stop is not None and stop != base_len) else base_len
            step = index.step if index.step is not None else ir.One
        else:
            # first dim removed
            return ir.SingleDimRef(node.value, dim=ir.One)
    elif isinstance(node, ir.NameRef):
        return ir.SingleDimRef(node, dim=ir.Zero)
    elif isinstance(node, ir.AffineSeq):
        start = node.start
        stop = node.stop
        step = node.step
    else:
        msg = f'No method to compute element count for {node}.'
        raise CompilerError(msg)

    if start == ir.Zero:
        diff = stop
    else:
        # It's safer to put the max around start, since this doesn't risk overflow
        diff = ir.SUB(stop, ir.MAX(ir.Zero, start))
    if step == ir.One:
        return diff
    base_count = ir.FLOORDIV(diff, step)
    test = ir.MOD(diff, step)
    fringe = ir.SELECT(predicate=test, on_true=ir.One, on_false=ir.Zero)
    count = ir.ADD(base_count, fringe)
    return count
