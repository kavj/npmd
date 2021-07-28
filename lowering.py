import numbers
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


def rewrite_if_matches_xor(expr):
    # Python's ast can't directly translate xor
    # but it's provided by lower level interfaces.

    # Capture simple cases that are either obvious or
    # follow from DeMorgan's Laws

    pass


def is_pow(expr):
    return isinstance(expr, ir.BinOp) and expr.op in ("**", "**=")


def rewrite_if_matches_square_root(expr):
    if is_pow(expr):
        repl = expr
        coeff = expr.right
        if coeff == ir.FloatConst(0.5):
            # Todo: needs a namespace or direct specialization
            repl = ir.Call("sqrt", (expr.left), ())
    return repl


def rewrite_if_matches_square(expr):
    repl = expr
    if is_pow(expr):
        coeff = expr.right
        if (coeff == ir.IntConst(2)) or (coeff == ir.FloatConst(2.0)):
            op = "*=" if expr.in_place else "*"
            repl = ir.BinOp(expr.left, expr.left, op)
    return repl


def rewrite_if_matches_sign_flip(expr):
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


def simplify_binop(expr: ir.BinOp):
    repl = rewrite_if_matches_square(expr)
    repl = rewrite_if_matches_square_root(repl)
    repl = rewrite_if_matches_sign_flip(repl)
    if repl == expr:
        repl = expr
    return repl


class ConstFolder(ExpressionVisitor):

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


def discard_unbounded(iterables):
    bounded = {it for it in iterables if not (isinstance(it, ir.AffineSeq) and it.stop is None)}
    return bounded


@singledispatch
def split_intervals(iterable):
    msg = f"No method to make counter for {iterable}."
    raise NotImplementedError(msg)


@split_intervals.register
def _(iterable: ir.NameRef):
    return ((ir.Zero, ir.Length(iterable), ir.One),)


@split_intervals.register
def _(iterable: ir.AffineSeq):
    return ((iterable.start, iterable.stop, iterable.step))


@split_intervals.register
def _(iterable: ir.Subscript):
    if isinstance(iterable.slice, ir.Slice):
        # We can convert iteration over a sliced array
        # to affine parameters with respect to the base array.
        start = iterable.start
        step = iterable.step
        base = (start, ir.Length(iterable.value), step)
        if iterble.stop is None:
            intervals = (base,)
        else:
            intervals = base, (start, iterable.stop, step)
    else:
        intervals = ((ir.Zero, ir.Length(iterable), ir.One),)
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

    start_count = len(starts)
    stop_count = len(stops)

    if start_count == stop_count == 1:
        start, = starts
        stop, = stops
    elif start_count == 1:
        start, = starts
        stop = ir.Min(tuple(stops))
    elif stop_count == 1:
        start = ir.Max(tuple(starts))
        stop, = stops
    else:
        widths = set()
        for start, stop in start_stop:
            if start == ir.Zero:
                widths.add(stop)
            else:
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
            if start == ir.Zero:
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


def make_min_sliced_len_expr(slices, leading_dim_expr):
    if not slices:
        return leading_dim_expr
    s = slices[0]
    bound = leading_dim_expr if (s.stop is None or s.start == ir.Zero) else ir.BinOp(s.stop, s.start, "-")
    for s in slices[1:]:
        b = leading_dim_expr if (s.stop is None or s.start == ir.Zero) else ir.BinOp(s.stop, s.start, "-")
        bound = ir.Ternary(ir.BinOp(bound, b, "<"), bound, b)
    return bound


def make_explicit_iter_count(counter):
    if all(isinstance(subexpr, ir.IntConst) for subexpr in counter.subexprs):
        count = numeric_max_iter_count(counter)
    elif counter.step == ir.One:
        count = ir.BinOp(counter.stop, counter.start, "-")
    else:
        on_false = ir.BinOp(counter.stop, counter.start, "-")
        # avoid integer division
        test = ir.BinOp(interval, counter.step, "&")
        on_true = ir.BinOp(on_false, ir.One, "+")
        count = ir.Ternary(test, on_true, on_false)
    return count

