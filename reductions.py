import itertools
import math
import typing

import ir

from symbol_table import symbol_table
from type_checks import TypeHelper
from utils import extract_name


class ExpressionMapper:
    """
    This handles labeling of expressions.

    """
    def __init__(self, symbols: symbol_table):
        self.symbols = symbols
        self.type_helper = TypeHelper(symbols)
        self.mapped = {}

    def is_mapped(self, node: ir.Expression):
        """
        determines whether an expression has been seen
        :param node:
        :return:
        """
        return node in self.mapped

    def map_expr(self, node: ir.ValueRef, target: typing.Optional[ir.NameRef] = None):
        # should be careful about caching subscripts
        if not isinstance(node, ir.ValueRef):
            msg = f"Node '{node}' is not a value reference."
            raise TypeError(msg)
        if not isinstance(node, ir.Expression):
            return node
        name = self.mapped.get(node)
        if name is None:
            if target is None:
                target_name = extract_name(target)
            else:
                target_name = "t"
            t = self.type_helper.check_type(node)
            name = self.symbols.make_unique_name_like(target_name, t)
            self.mapped[node] = name
        return name


def flatten_min_max_reduction(node: typing.Union[ir.MinReduction, ir.MaxReduction]):
    """
    Useful for optimizing integer reductions in cases of
    :param node:
    :return:
    """
    assert isinstance(node, (ir.MinReduction, ir.MaxReduction))
    flatten_types = (ir.Min, ir.MinReduction) if isinstance(node, ir.MaxReduction) else (ir.Max, ir.MaxReduction)
    terms = set()
    queued = []
    for term in node.subexprs:
        if isinstance(term, flatten_types):
            queued.extend(term.subexprs)
        else:
            terms.add(term)
    while queued:
        term = queued.pop()
        if isinstance(term, flatten_types):
            queued.extend(term.subexprs)
        else:
            terms.add(term)
    terms = list(terms)
    return terms


def simple_serialize_min_max(node: typing.Union[ir.MinReduction, ir.MaxReduction]):
    """
    Min max serialize without scraping all pairs
    :param node:
    :return:
    """
    if isinstance(node, ir.MinReduction):
        reducer = ir.Min
    elif isinstance(node, ir.MaxReduction):
        reducer = ir.Max
    else:
        msg = f"serializer requires min or max reduction. Received {node}."
        raise TypeError(msg)

    terms = flatten_min_max_reduction(node)

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
