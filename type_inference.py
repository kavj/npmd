import itertools
import textwrap

from collections import defaultdict
from contextlib import contextmanager
from functools import singledispatchmethod

import ir
import utils

import pretty_printing as pp
import type_resolution as tr

from errors import CompilerError
from visitor import StmtVisitor, ExpressionVisitor


class TypeMismatch:

    def __init__(self, msg):
        self.msg = msg


class ExprTypeInfer(ExpressionVisitor):
    """
    Intentionally basic implementation of the Cartesian Product Algorithm.
    Everything eventually has to be resolvable to a fixed data type without promoting
    non-constant integers to floating point values or similar.

    As an example, most people will initialize "a" to "1" this way regardless of whether "a"
    is used as an integer or floating point type past that point. In spite of this, the type of "c"
    is unambiguous. In that sense, it should be okay to allow the use of a product like this as long
    as it doesn't introduce type ambiguity into any downstream use, with use determined by variable name,
    rather than explicit dataflow analysis.

    a = 1     # possible types (int, float)
    b = 2.5   # possible types (float)

    c = a * b  # possible expression signatures: (int, float) -> float, (float, float) -> float

    references:
    Ole Ageson, The Cartesian Product Algorithm
    Simple and Precise Type Inference of Parametric Polymorphism

    """

    def __init__(self, types):
        # types updated externally
        self._types = types
        # track what exprs use each parameter and vice-versa, so as to cascade updates
        # so we really need an ordered set here for safe dependency ordering
        # for now, try using ordered properties of dictionary key insertions in recent python versions
        self._used_by = defaultdict(dict)
        self.format_expr = pp.pretty_formatter()

    def __call__(self, expr):
        assert isinstance(expr, ir.ValueRef)
        return self.visit(expr)

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
                    return TypeMismatch(msg)
                api_sig.append(api_type)
            api_sigs.append(api_sig)
        formatted_sigs = ", ".join(str(sig) for sig in api_sigs)
        return TypeMismatch(formatted_expr + formatted_sigs)

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Subscript):
        type_ = self._types.get(node.value)
        if isinstance(type_, ir.ArrayType):
            # would be better not to raise here..
            msg = f"Cannot subscript non-array type {type_}."
            raise CompilerError(msg)
        dtype = type_.dtype

        if not isinstance(node.slice, ir.Slice):
            ndims = type_.ndims - 1
            if ndims == 0:
                type_ = dtype
            else:
                type_ = ir.ArrayType(ndims, dtype)
        return type_

    @visit.register
    def _(self, node: ir.NameRef):
        return self._types.get(node)

    @visit.register
    def _(self, node: ir.BinOp):
        # no caching since they may change
        left = self.visit(node.left)
        right = self.visit(node.right)
        for types in itertools.chain(left, right):
            if isinstance(types, TypeMismatch):
                return types
        possible_types = defaultdict(set)
        candidates = []
        for pair in itertools.product(left, right):
            candidates.append(pair)
            match = tr.binops_dispatch.get(pair)
            if match is not None:
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
        return tuple(possible_types.keys())

    @visit.register
    def _(self, node: ir.UnaryOp):
        return self.visit(node.operand)

    @visit.register
    def _(self, node: ir.CompareOp):
        # Todo: currently assumes all scalar types... need to generalize
        left = self.visit(node.left)
        right = self.visit(node.right)
        output_types = []
        for ltype, rtype in itertools.product(left, right):
            output_types.append(tr.merge_truth_types((ltype, rtype)))
            max_bit_width = max(ltype.bits, rtype.bits)
            is_integral = ltype.integral and rtype.integral
            t = tr.type_from_spec(max_bit_width, is_integral, is_boolean=True)
            output_types.append(t)
        return tuple(output_types)

    @visit.register
    def _(self, node: ir.BoolOp):
        input_types = [self.visit(operand) for operand in node.subexprs]
        output_types = []
        for types in itertools.product(*input_types):
            output_types.append(tr.merge_truth_types(types))
        return tuple(output_types)


class IteratedExprInfer(ExpressionVisitor):
    """
    Infer type for an expression that is directly iterated over
    """

    def __init__(self, types):
        self.expr_typer = ExprTypeInfer(types)
        self.types = types

    @singledispatchmethod
    def visit(self, expr):
        msg = f"No handler for type {type(expr)}"
        return TypeMismatch(msg)

    @visit.register
    def _(self, expr: ir.AffineSeq):
        start = self.expr_typer.visit(expr.start)
        stop = self.expr_typer.visit(expr.stop)
        step = self.expr_typer.visit(expr.step)
        for label, term in zip(("start", "stop", "step"), (start, stop, step)):
            if not isinstance(term, ir.ScalarType) or not term.integral:
                msg = f"Parameter {label} must have integer type."
                return TypeMismatch(msg)
        # should add standard integer lookup here
        return tr.Int64

    @visit.register
    def _(self, expr: ir.NameRef):
        base_type = self.types.get(expr)
        if isinstance(base_type, TypeMismatch):
            return base_type
        elif isinstance(base_type, ir.ArrayType):
            ndims = base_type.ndims - 1
            dtype = base_type.dtype
            if ndims == 0:
                return dtype
            else:
                return ir.ArrayType(ndims, dtype)
        elif isinstance(base_type, ir.AffineSeq):
            return tr.Int64
        else:
            msg = f"Cannot iterate over type {base_type}."
            return TypeMismatch(msg)

    @visit.register
    def _(self, expr: ir.Subscript):
        base_type = self.expr_typer.visit(expr.value)
        if isinstance(base_type, ir.ArrayType):
            if isinstance(expr.slice, ir.Slice):
                return base_type
            index_type = self.expr_typer.visit(expr.slice)
            if isinstance(index_type, ir.ScalarType):
                if index_type.integral:
                    ndims = base_type.ndims - 1
                    dtype = base_type.dtype
                    if ndims == 0:
                        return dtype
                    else:
                        return ir.ArrayType(ndims, dtype)
        else:
            msg = f"{type(expr)} cannot be subscripted."
            return TypeMismatch(msg)

    @visit.register
    def _(self, expr: ir.BinOp):
        assert not expr.in_place
        ltype = self.visit(expr.left)
        rtype = self.visit(expr.right)
        # this needs shared parameter info

    @visit.register
    def _(self, expr: ir.UnaryOp):
        pass


class TypeAssign(StmtVisitor):

    def __init__(self, types):
        self.assigned = []
        self.types = types
        self.expr_typer = ExprTypeInfer(types)
        self.iterated_typer = IteratedExprInfer(types)
        self.loop = None

    def __call__(self, types):
        pass

    @contextmanager
    def enclosing_loop(self, header):
        assert isinstance(header, (ir.ForLoop, ir.WhileLoop))
        stashed = self.loop
        self.loop = header
        yield
        enclosing = self.loop
        assert enclosing is header
        self.loop = stashed

    @singledispatchmethod
    def visit(self, stmt):
        return super().visit(stmt)

    @visit.register
    def _(self, stmt: ir.Assign):
        # If name target, check type of rhs
        if isinstance(stmt.target, ir.NameRef):
            value_type = self.visit(stmt.value)
            self.types[stmt.target].add(value_type)

    @visit.register
    def _(self, stmt: ir.ForLoop):
        for target, iterable in utils.unpack_iterated(stmt.target, stmt.iterable, True):
            if not isinstance(target, ir.NameRef):
                continue
            type_ = self.iterated_typer.visit(iterable)
            if isinstance(type_, TypeMismatch):
                return type_
            if not isinstance(type_, (ir.ArrayType, ir.AffineSeq)):
                self.types[target].add(iterable)

    @visit.register
    def _(self, stmt: ir.IfElse):
        test_type = self.visit(stmt.test)
        try:
            # Todo: This could return None on no type match,
            #   since the function is too low level to
            #   produce meaningful error messages.
            tr.truth_type_from_type(test_type)
        except TypeError:
            interface_type = pp.get_pretty_type(test_type)
            msg = f"Cannot truth cast {interface_type}."
            raise CompilerError(msg)
        self.visit(stmt.if_branch)
        self.visit(stmt.else_branch)
