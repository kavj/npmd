import numbers
import math
import operator
import typing

from collections import defaultdict
from dataclasses import dataclass
from functools import singledispatch, singledispatchmethod

import ir
import symbol_table

from visitor import StmtVisitor, ExpressionVisitor


unaryops = {"+": operator.pos,
            "-": operator.neg,
            "~": operator.inv,
            "not": operator.not_,
            }

binops = {"+": operator.add,
          "-": operator.sub,
          "*": operator.mul,
          "/": operator.truediv,
          "//": operator.floordiv,
          "%": operator.mod,
          "**": operator.pow,
          "@": operator.matmul,
          "+=": operator.iadd,
          "-=": operator.isub,
          "*=": operator.imul,
          "/=": operator.ifloordiv,
          "//=": operator.itruediv,
          "%=": operator.imod,
          "**=": operator.ipow,
          "@=": operator.imatmul,
          "==": operator.eq,
          "!=": operator.ne,
          "<": operator.lt,
          "<=": operator.le,
          ">": operator.gt,
          ">=": operator.ge,
          "<<": operator.lshift,
          ">>": operator.rshift,
          "&": operator.and_,
          "|": operator.or_,
          "^": operator.xor,
          "<<=": operator.ilshift,
          ">>=": operator.irshift,
          "&=": operator.iand,
          "|=": operator.ior,
          "^=": operator.ixor,
          "isnot": NotImplemented,
          "in": NotImplemented,
          "notin": NotImplemented
          }


def wrap_constant(c):
    if isinstance(c, bool):
        return ir.BoolConst(c)
    if isinstance(c, numbers.Integral):
        return ir.IntConst(c)
    elif isinstance(c, numbers.Real):
        return ir.FloatConst(c)
    else:
        msg = f"Can't construct constant node for unsupported constant type {type(c)}"
        raise NotImplementedError(msg)


def is_pow(expr):
    return isinstance(expr, ir.BinOp) and expr.op in ("**", "**=")


def rewrite_pow(expr):
    if not is_pow(expr):
        return expr
    coeff = expr.right
    if coeff == ir.Zero:
        return ir.One
    elif coeff == ir.One:
        return expr.left
    elif coeff == ir.IntConst(-1):
        op = "/=" if expr.in_place else "/"
        return ir.BinOp(ir.One, expr.left, op)
    elif coeff == ir.IntConst(-2):
        op = "/=" if expr.in_place else "/"
        return ir.BinOp(ir.One, ir.BinOp(expr.left, expr.left, "*"), op)
    elif coeff == ir.IntConst(2):
        op = "*=" if expr.in_place else "*"
        return ir.BinOp(expr.left, expr.left, op)
    elif coeff == ir.FloatConst(0.5):
        return ir.Call("sqrt", (expr.left,), ())
    else:
        return expr


class simplify_exprs(ExpressionVisitor):
    """
    This class helps clean up composite adjustments made by other visitors.
    Placing it here helps omit having to perform an excessive number of checks in each
    individual expression rewrite. This is meant to run on expression hierarchies prior
    to scheduling the introduction of temporary variables.

    This contains some parts for associative ops, which should be disabled with floating point types.

    """

    def __init__(self, syms):
        self.syms = syms

    def __call__(self, expr):
        return self.lookup(expr)

    @singledispatchmethod
    def visit(self, expr):
        super().visit(expr)

    @visit.register
    def _(self, expr: ir.BinOp):
        if is_pow(expr):
            expr = rewrite_pow(expr)
        left = fold_identity(expr.left)
        right = fold_identity(expr.right)
        op = expr.op
        if left.constant and right.constant:
            oper = binops[op]
            value = oper(left.value, right.value)
            value = wrap_constant(value)
        if op in ("*", "*="):
            if left == ir.One:
                return right
            elif right == ir.One:
                return left
            elif left == ir.Zero or right == ir.Zero:
                return ir.Zero
            elif left == ir.IntConst(-1):
                return ir.UnaryOp(right, "-")
            elif right == ir.IntConst(-1):
                return ir.UnaryOp(left, "-")
        elif op in ("+", "+="):
            if left == ir.Zero:
                return right
            elif right == ir.Zero:
                return left
        elif op in ("-", "-="):
            if left == ir.Zero:
                return ir.UnaryOp(right, "-")
            elif right == ir.Zero:
                return left
            elif left == right:
                return ir.Zero
        elif op in ("//", "//="):
            if right == ir.One:
                return left
        else:
            return expr

    @visit.register
    def _(expr: ir.UnaryOp):
        if op == "+":
            return operand
        elif op == "-":
            if isinstance(operand, ir.UnaryOp) and operand.op == "-":
                return operand.operand
        else:
            return expr

        repl = expr
        if isinstance(expr, ir.BinOp) and op in ("*", "*="):
            unary_minus_equiv = ir.IntConst(-1)
            left = expr.left
            right = expr.right
            if left == unary_minus_equiv:
                repl = ir.UnaryOp(right, "-")
            elif right == unary_minus_equiv:
                repl = ir.UnaryOp(left, "-")
        return repl

    @visit.register
    def _(expr: ir.Ternary):
        # Fold if both sides match
        # Fold to min or max if matches ordered form
        # a few others
        pass

    @visit.register
    def _(expr: ir.Max):
        # remove nesting and duplicates
        repl = set()
        numeric = None
        contains_nan = False
        contains_fp_consts = False
        contains_int_consts = False
        nested_mins = []
        # Todo: need a flattening step for nested max exprs
        for elem in expr.exprs:
            if elem.constant and not contains_nan:
                if math.isnan(elem.value):
                    contains_nan = True
                    numeric = math.nan
                elif numeric is None:
                    numeric = elem.value
                else:
                    numeric = max(numeric, elem.value)
            elif isinstance(elem, ir.Max):
                repl.update(elem.exprs)
            else:
                repl.append(elem)
        return


    @visit.register
    def _(expr: ir.Min):
        pass


def simplify_binop(expr: ir.BinOp):
    repl = rewrite_if_matches_square(expr)
    repl = rewrite_if_matches_square_root(repl)
    repl = rewrite_if_matches_sign_flip(repl)
    if repl == expr:
        repl = expr
    return repl


class const_folding(ExpressionVisitor):

    def __call__(self, expr):
        return self.lookup(expr)

    @singledispatchmethod
    def visit(self, expr):
        super().visit(expr)

    @visit.register
    def _(self, expr: ir.BinOp):
        left = self.lookup(expr.left)
        right = self.lookup(expr.right)
        if left.constant and right.constant:
            op = binops[expr.op]
            value = op(left.value, right.value)
            result = wrap_constant(value)
            return result
        else:
            return expr

    @visit.register
    def _(self, expr: ir.UnaryOp):
        operand = self.lookup(expr.operand)
        if operand.constant:
            op = unaryops[expr.op]
            value = op(operand.value)
            result = wrap_constant(value)
            return result
        else:
            return expr

    @visit.register
    def _(self, expr: ir.Ternary):
        test = self.lookup(expr.test)
        if_expr = self.lookup(expr.if_expr)
        else_expr = self.lookup(expr.else_expr)
        if test.constant:
            return if_expr if operator.truth(test) else else_expr
        else:
            return ir.Ternary(test, if_expr, else_expr)

    @visit.register
    def _(self, expr: ir.AND):
        operands = []
        for operand in self.operands:
            operand = self.lookup(operand)
            if operand.constant:
                if not operator.truth(operand):
                    return ir.BoolConst(False)
            else:
                operands.append(operand)
        if len(operands) >= 2:
            operands = tuple(operands)
            return ir.AND(operands)
        else:
            operands = operands.pop()
            return ir.TRUTH(operands)

    @visit.register
    def _(self, expr: ir.OR):
        operands = []
        for operand in self.operands:
            operand = self.lookup(operand)
            if operand.constant:
                if operator.truth(operand):
                    return ir.BoolConst(True)
            else:
                operands.append(operand)
        if len(operands) >= 2:
            operands = tuple(operands)
            return ir.OR(operands)
        else:
            operands = operands.pop()
            return ir.TRUTH(operands)

    @visit.register
    def _(self, expr: ir.XOR):
        # not sure about this one, since
        # it's not monotonic
        raise NotImplementedError

    @visit.register
    def _(self, expr: ir.Max):
        raise NotImplementedError

    @visit.register
    def _(self, expr: ir.Min):
        raise NotImplementedError


def discard_unbounded(iterables):
    bounded = {it for it in iterables if not (isinstance(it, ir.AffineSeq) and it.stop is None)}
    return bounded


class interval_splitting(ExpressionVisitor):

    @singledispatchmethod
    def visit(iterable):
        msg = f"No method to make counter for {iterable}."
        raise NotImplementedError(msg)

    @visit.register
    def _(iterable: ir.NameRef):
        return {(ir.Zero, ir.Length(iterable), ir.One)}

    @visit.register
    def _(iterable: ir.AffineSeq):
        # Note: we have to return an interval even if the stop parameter is None
        # Without this, the
        return {(iterable.start, iterable.stop, iterable.step)}

    @visit.register
    def _(iterable: ir.Subscript):
        if isinstance(iterable.slice, ir.Slice):
            # We can convert iteration over a sliced array
            # to affine parameters with respect to the base array.
            start = iterable.start
            step = iterable.step
            base = (start, ir.Length(iterable.value), step)
            if iterble.stop is None:
                intervals = {base}
            else:
                intervals = {base, (start, iterable.stop, step)}
        else:
            intervals = {(ir.Zero, ir.Length(iterable), ir.One),}
        return intervals


def get_sequence_step(iterable):
    if isinstance(iterable, ir.Subscript):
        if isinstance(iterable.slice, ir.Slice):
            return iterable.slice.step
    elif isinstance(iterable, ir.AffineSeq):
        return iterable.step
    return ir.One


def find_min_interval(spans):
    starts = set()
    stops = set()
    for start, stop in spans:
        starts.add(start)
        stops.add(stop)

    # ignore unbounded
    stops.discard(None)

    start_count = len(starts)
    stop_count = len(stops)

    if start_count == 1 and stop_count <= 1:
        start, = starts
        if stops:
            stop, = stops
        else:
            stop = None
    elif start_count == 1:
        start, = starts
        stop = ir.Min(tuple(stops)) if stops else None
    elif stop_count <= 1:
        start = ir.Max(tuple(starts))
        if stops:
            stop, = stops
        else:
            stop = None
    else:
        widths = set()
        for start, stop in start_stop:
            if start == ir.Zero:
                widths.add(stop)
            elif stop is not None:
                d = ir.BinOp(stop, start, "-")
                widths.add(d)
        start = ir.Zero
        stop = ir.Min(tuple(spans))
    return start, stop


def make_loop_counter(iterables, syms):
    """
    Map a combination of array iterators and range and enumerate calls to a set of counters,
    which capture the appropriate intervals.

    In the case of arrays, this refers to the possibly non-linear
    index into the array at each step, which should be taken by the iterator at runtime.

    """

    intervals = set()
    split_intervals = interval_splitting.lookup

    for iterable in iterables:
        from_iterable = split_intervals(iterable)
        intervals.update(from_iterable)

    by_step = defaultdict(set)

    for interval in intervals:
        (start, stop, step) = interval
        ss = (start, stop)
        by_step[step].add(ss)

    # If we have one step size
    # make it the loop variable step

    intervals = {}

    # reduce to one interval per step
    for step, start_stop in by_step.items():
        start, stop = find_min_interval(start_stop)
        intervals[step] = (start, stop)

    if len(intervals) == 1:
        step, (start, stop) = intervals.popitem()
    else:
        # compute interval count explicitly
        reduce = set()
        for step, (start, stop) in intervals:
            if stop is None:
                # This still requires a compatible step,
                # but it doesn't bound anything.
                continue
            elif start == ir.Zero:
                diff = stop
            else:
                diff = ir.BinOp(stop, start, "-")
            base_count = ir.BinOp(diff, step, "//")
            rem = ir.BinOp(diff, step, "&")
            fringe = ir.Ternary(ir.BinOp(rem, ir.Zero, ">"), ir.One, ir.Zero)
            count = ir.BinOp(base_count, fringe, "+")
            reduce.add(count)
        start = ir.Zero
        stop = ir.Min(tuple(reduce))
        step = ir.One
    counter = ir.AffineSeq(start, stop, step)

    return counter
