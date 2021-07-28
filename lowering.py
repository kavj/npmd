import numbers
import operator
import typing

from collections import defaultdict
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
def make_affine_counter(iterable, symbols):
    msg = f"No method to make counter for {iterable}."
    raise NotImplementedError(msg)


@make_affine_counter.register
def _(iterable: ir.NameRef):
    counter = ir.AffineSeq(ir.Zero, ir.Length(iterable), ir.One)
    return counter


@make_affine_counter.register
def _(iterable: ir.Subscript):
    if isinstance(iterable.slice, ir.Slice):
        # We can convert iteration over a sliced array
        # to affine parameters with respect to the base array.
        start = iterable.start
        stop = ir.Length(iterable.value)
        step = iterable.step
        if iterable.stop is not None:
            stop = ir.Min(stop, iterable.stop)
    else:
        start = ir.Zero
        stop = ir.Length(iterable)
        step = ir.One
    counter = ir.AffineSeq(start, stop, step)
    return counter


def get_sequence_step(iterable):
    if isinstance(iterable, ir.Subscript):
        if isinstance(iterable.slice, ir.Slice):
            return iterable.slice.step
    elif isinstance(iterable, ir.AffineSeq):
        return iterable.step
    return ir.One


# Todo: fill in stubs. These are needed to avoid sensitivity in ordering of
#       min/max expression creation for integer expressions.
def merge_integer_min(a: ir.Min, b: ir.Min):
    raise NotImplementedError


def merge_integer_max():
    raise NotImplementedError


def find_min_interval_width(intervals):
    lower_bounds = set()
    upper_bounds = set()
    fold_if_constant = ConstFolder()

    for lower, upper in intervals:
        lower_bounds.add(lower)
        upper_bounds.add(upper)

    unique_upper_count = len(upper_bounds)
    unique_lower_count = len(lower_bounds)

    if unique_upper_count == unique_lower_count == 1:
        lower, = lower_bounds
        upper, = upper_bounds
        diff_expr = ir.BinOp(upper, lower, "-")
        min_interval_width = fold_if_constant(expr)
        assert isinstance(min_interval_width, ir.IntConst) or not min_interval_width.constant
    elif unique_upper_count == 1:
        lower = ir.Max(lower_bounds)
        upper, = upper_bounds
        min_interval_width = ir.BinOp(upper, lower, "-")
    elif unique_lower_count == 1:
        lower, = lower_bounds
        upper = ir.Min(upper_bounds)
        min_interval_width = ir.BinOp(upper, lower, "-")
    else:
        # reduce over all pairs
        interval_widths = set()
        numeric_min = None
        for lower, upper in intervals:
            width = ir.BinOp(upper, lower, "-")
            width = fold_if_constant(width)
            if width.constant:
                assert isinstance(width, ir.IntConst)
                if numeric_min is None:
                    numeric_min = width.value
                else:
                    numeric_min = min(width.value, numeric_min)
            else:
                interval_widths.add(width)
        if numeric_min is None:
            min_interval_width = ir.Min(interval_widths)
        elif numeric_min <= 0:
            # avoid clamping here
            min_interval_width = wrap_constant(numeric_min)
        else:
            interval_widths.add(wrap_constant(numeric_min))
            min_interval_width = ir.Min(interval_widths)
    return min_interval_width


def make_iter_count_expr(span, step):
    if step == ir.Zero:
        # catches some cases which are not folded, with a more detailed message
        msg = f"Interval or range calculations may not use a step size of zero."
        raise ZeroDivisionError(msg)
    if span.constant and step.constant:
        assert isinstance(span, ir.IntConst)
        assert isinstance(step, ir.IntConst)
        width = span.value
        step_ = step.value
        base_iteration_count = operator.floordiv(width, step_)
        fringe = 1 if operator.and_(width, step_) else 0
        iter_count = operator.add(base_iteration_count, fringe)
        iter_count = wrap_constant(iter_count)
    else:
        base_iteation_count = ir.BinOp(span, step, "//")
        test = ir.BinOp(span, step, "&")
        fringe = ir.Ternary(test, ir.One, ir.Zero)
        iter_count = ir.BinOp(base_iteation_count, fringe, "+")
    return iter_count


def make_loop_counter(iterables, syms):
    """
    Map a combination of array iterators and range and enumerate calls to a set of counters,
    which capture the appropriate intervals.

    In the case of arrays, this refers to the possibly non-linear
    index into the array at each step, which should be taken by the iterator at runtime.

    """

    counters = {make_affine_counter(iterable) for iterable in iterables}

    # check for simple case first
    if len(counters) == 1:
        counter, = counters
        return counter

    # group by step
    by_step = defaultdict(set)
    for c in counters:
        by_step[c.step].add((c.start, c.stop))

    # For each possibly unique step size, simplify runtime interval width
    # calculations as much as possible.
    min_interval_widths = {}
    for step, counter_group in by_step.items():
        min_interval_widths[step] = find_minimum_interval_width(counter_group)

    if len(min_interval_widths) == 1:
        # we can make a loop counter of the form range(0, min_interval_width, step)
        step, min_interval_width = min_interval_widths.popitem()
        counter = ir.AffineSeq(ir.Zero, min_interval_width, step)
    else:
        iter_counts = {make_iter_count_expr(span, step) for (span, step) in min_interval_widths.items()}
        min_iter_count = ir.Min(iter_counts)
        counter = ir.AffineSeq(ir.Zero, min_iter_count, ir.One)

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


def make_loop_interval(targets, iterables, symbols, loop_index):
    counters = {make_affine_counter(iterable, symbols) for iterable in iterables}
    # This has the annoying effect of dragging around the symbol table.
    # Since we need to be able to reach an array definition along each path
    # and array names are not meant to be reassigned, we should be passing an array
    # reference here.
    # counter = merge_loop_counters(counters, symbols, loop_index)
    return counters
