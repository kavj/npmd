import itertools
import math
import typing

import ir

from symbol_table import symbol_table
from type_checks import TypeHelper


class ExpressionMapper:
    def __init__(self, symbols: symbol_table):
        self.symbols = symbols
        self.type_helper = TypeHelper(symbols)
        self.mapped = {}
        self.assigns = []

    def is_mapped(self, node: ir.Expression):
        return node in self.mapped

    def map_expr(self, node: ir.ValueRef):
        if not isinstance(node, ir.ValueRef):
            msg = f"Node '{node}' is not a value reference."
            raise TypeError(msg)
        if not isinstance(node, ir.Expression):
            return node
        name = self.mapped.get(node)
        if name is None:
            t = self.type_helper.check_type(node)
            name = self.symbols.make_unique_name_like("i", t)
            self.mapped[node] = name
        self.assigns.append((name, node))
        return name

    def map_terms(self, nodes: typing.Iterable[ir.Expression]):
        for node in nodes:
            mapped = self.map_expr(node)
            yield mapped


def simple_serialize_min(node: typing.Union[ir.MinReduction, ir.MaxReduction]):
    terms = list(node.subexprs)
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
        terms = [ir.Min(left, right) for left, right
                 in zip(itertools.islice(terms, 0, None, 2), itertools.islice(terms, 1, None, 2))]
    assert len(terms) == 1
    reduced = terms[0]
    if tail is not None:
        reduced = ir.Min(reduced, tail)
    return reduced


def simple_serialize_max(node: typing.Union[ir.MinReduction, ir.MaxReduction]):

    terms = list(node.subexprs)
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
        terms = [ir.Max(left, right) for left, right
                 in zip(itertools.islice(terms, 0, None, 2), itertools.islice(terms, 1, None, 2))]
    assert len(terms) == 1
    reduced = terms[0]
    if tail is not None:
        reduced = ir.Max(reduced, tail)
    return reduced

