import itertools
import textwrap

from collections import defaultdict
from contextlib import contextmanager
from functools import singledispatchmethod

import ir

import pretty_printing as pp
import type_resolution as tr

from errors import CompilerError
from visitor import StmtVisitor, ExpressionVisitor


class ExprInfer(ExpressionVisitor):
    """
    Intentionally basic implementation of the Cartesian Product Algorithm.

    The reasoning here is pretty simple.
    Some cases, such as constants may be slightly ambiguous.

    For example, to initialize a variable "a" to 1, most people would write
    "a = 1", without considering whether it should be an integer or floating point value.
    Allowing parametric polymorphism here via argument templating further complicates this,
    as it reduces the cases where type annotations or typing "a = 1.0" for floating point values
    would fix this.

    Instead, we want to check that for any tuple of arguments formed from possible argument types,
    the resulting expression has an unambiguous type.

    As an example

    a = 1     # possible types (int, float)
    b = 2.5   # possible types (float)

    c = a * b  # possible expression signatures: (int, float) -> float, (float, float) -> float

    references:
    Ole Ageson, The Cartesian Product Algorithm
    Simple and Precise Type Inference of Parametric Polymorphism

    """

    def __init__(self, types, disallowed):
        # types updated externally
        self._types = types
        self._disallowed = disallowed
        # track what exprs use each parameter and vice-versa, so as to cascade updates
        # so we really need an ordered set here for safe dependency ordering
        # for now, try using ordered properties of dictionary key insertions in recent python versions
        self._used_by = defaultdict(dict)
        self.format_expr = pp.pretty_formatter()

    def format_type_error(self, expr, type_sigs):
        """
        Format type mismatches discovered internally to something corresponding to numpy api types.
        """
        formatted_expr = self.format_expr(expr)
        api_sigs = []
        for sig in type_sigs:
            api_sig = []
            for param in sig:
                api_type = pp.get_pretty_type(param)
                if api_type is None:
                    msg = f"Unrecognized implementation type {param}."
                    raise TypeError(msg)
                api_sig.append(api_type)
            api_sigs.append(api_sig)
        formatted_sigs = ", ".join(str(sig) for sig in api_sigs)
        return expr, formatted_sigs

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.BinOp):
        # no caching since they may change
        left = self.visit(node.left)
        right = self.visit(node.right)
        possible_types = defaultdict(set)
        candidates = []
        disallowed = self._disallowed.get(node, ())
        for pair in itertools.product(left, right):
            candidates.append(pair)
            match = tr.binops_dispatch.get(pair)
            if match is not None:
                if match not in disallowed:
                    possible_types[match].add(pair)
        # error if no signature is feasible
        if len(possible_types) != 0:
            self._types[node] = possible_types
        else:
            formatted_expr, formatted_sigs = self.format_type_error(node, candidates)
            # there is a remote possibility this could produce a long list
            msg = f"No candidate signature matches expression {formatted_expr}. Candidate signatures are" \
                  f"{formatted_sigs}."
            msg = textwrap.wrap(msg)
            raise CompilerError(msg)
        return possible_types


class TypingVisitor(StmtVisitor):

    def __init__(self, types):
        self.assigned = []
        self.types = types

    def __call__(self, types):
        pass

    @contextmanager
    def flow_region(self):
        pass

    @singledispatchmethod
    def visit(self, stmt):
        return super().visit(stmt)

    # Do functions belong here?

    @visit.register
    def _(self, stmt: ir.Assign):
        # If name target, check type of rhs
        pass

    @visit.register
    def _(self, stmt: ir.ForLoop):
        # check target assignments, update mapping
        # for nameref targets
        pass
