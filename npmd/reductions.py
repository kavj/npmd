import itertools
import math

from functools import singledispatchmethod
from typing import Iterable, List, Union

import npmd.ir as ir

from npmd.symbol_table import SymbolTable
from npmd.type_checks import TypeHelper, is_integer


def unpack_min_terms(terms: Iterable[ir.ValueRef], type_check: TypeHelper) -> List[ir.ValueRef]:
    repl = []
    queued = [*terms]
    seen = set()
    while queued:
        term = queued.pop()
        if term in seen:
            continue
        seen.add(term)
        if isinstance(term, (ir.MIN, ir.MinReduction)):
            # safety check needed so we don't reorder
            if all(is_integer(type_check(subexpr)) for subexpr in term.subexprs):
                queued.extend(term.subexprs)
            else:
                repl.append(term)
        else:
            repl.append(term)
    return repl


def unpack_max_terms(terms: Iterable[ir.ValueRef], type_check: TypeHelper) -> List[ir.ValueRef]:
    repl = []
    queued = [*terms]
    seen = set()
    while queued:
        term = queued.pop()
        if term in seen:
            continue
        seen.add(term)
        if isinstance(term, (ir.MAX, ir.MaxReduction)):
            if all(is_integer(type_check(subexpr)) for subexpr in term.subexprs):
                queued.extend(term.subexprs)
            else:
                repl.append(term)
        else:
            repl.append(term)
    return repl


def wrap_max_reduction(terms: Iterable[ir.ValueRef], type_check: TypeHelper) -> ir.ValueRef:
    ordered = frozenset(unpack_max_terms(terms, type_check))
    if len(ordered) == 1:
        return next(iter(ordered))
    elif len(ordered) == 2:
        return ir.MAX(*ordered)
    else:
        return ir.MaxReduction(ordered)


def wrap_min_reduction(terms: List[ir.ValueRef], type_check: TypeHelper) -> ir.ValueRef:
    ordered = frozenset(unpack_min_terms(terms, type_check))
    if len(ordered) == 1:
        return next(iter(ordered))
    elif len(ordered) == 2:
        return ir.MIN(*ordered)
    else:
        return ir.MinReduction(*ordered)


class NormalizeMinMax:
    """
    Reorders in integer cases, where ordering is arbitrary. Also consolidates identical terms in such cases.
    """

    def __init__(self, symbols: SymbolTable):
        self.symbols = symbols
        self.typer = TypeHelper(symbols)

    def __call__(self, node: ir.ValueRef) -> ir.ValueRef:
        return self.visit(node)

    @singledispatchmethod
    def visit(self, node):
        msg = f'No method to apply normalize integral to "{node}".'
        raise TypeError(msg)

    @visit.register
    def _(self, node: ir.ValueRef):
        return node

    @visit.register
    def _(self, node: ir.Expression):
        repl = node.reconstruct(*(self.visit(subexpr) for subexpr in node.subexprs))
        return repl

    @visit.register
    def _(self, node: ir.MAX):
        repl = ir.MAX(*(self.visit(subexpr) for subexpr in node.subexprs))
        if any(not is_integer(self.typer(subexpr)) for subexpr in repl.subexprs):
            return repl
        return wrap_max_reduction(repl.subexprs, self.typer)

    @visit.register
    def _(self, node: ir.MIN):
        repl = ir.MIN(*(self.visit(subexpr) for subexpr in node.subexprs))
        if any(not is_integer(self.typer(subexpr)) for subexpr in repl.subexprs):
            return repl
        return wrap_min_reduction(repl.subexprs, self.typer)

    @visit.register
    def _(self, node: ir.MinReduction):
        # we don't pay attention to ordering on these
        repl = ir.MinReduction(*(self.visit(subexpr) for subexpr in node.subexprs))
        return wrap_min_reduction(repl.subexprs, self.typer)

    @visit.register
    def _(self, node: ir.MaxReduction):
        repl = ir.MaxReduction(*(self.visit(subexpr) for subexpr in node.subexprs))
        return wrap_max_reduction(repl.subexprs, self.typer)


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
