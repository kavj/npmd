import numbers
import operator
import typing

from collections import defaultdict
from functools import singledispatch, singledispatchmethod

import ir
import symbols

from visitor import StmtVisitor, walk, walk_branches


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


def is_pow(expr):
    return isinstance(expr, ir.BinOp) and expr.op in ("**", "**=")


def simplify_if_square_root(expr):
    if is_pow(expr):
        repl = expr
        coeff = expr.right
        if coeff == ir.FloatNode(0.5):
            # Todo: needs a namespace or direct specialization
            repl = ir.Call("sqrt", (expr.left), ())
    return repl


def simplify_if_is_square(expr):
    repl = expr
    if is_pow(expr):
        coeff = expr.right
        if (coeff == ir.IntNode(2)) or (coeff == ir.FloatNode(2.0)):
            op = "*=" if expr.in_place else "*"
            repl = ir.BinOp(expr.left, expr.left, op)
    return repl


def simplify_if_sign_flip(expr):
    repl = expr
    if isinstance(expr, ir.BinOp) and op in ("*", "*="):
        unary_minus_equiv = ir.IntNode(-1)
        left = expr.left
        right = expr.right
        if left == unary_minus_equiv:
            repl = ir.UnaryOp(right, "-")
        elif right == unary_minus_equiv:
            repl = ir.UnaryOp(left, "-")
    return repl


def simplify_binop(expr: ir.BinOp):
    repl = simplify_if_is_square(expr)
    repl = simplify_if_square_root(repl)
    repl = simplify_if_sign_flip(repl)
    return repl


@singledispatch
def fold_if_constant(expr):
    msg = f"fold expression not implemented for "
    raise NotImplementedError


@fold_if_constant.register
def _(expr: ir.Constant):
    return expr


@fold_if_constant.register
def _(expr: ir.NameRef):
    return expr


@fold_if_constant.register
def _(expr: ir.Slice):
    start = fold_if_constant(expr.start)
    stop = expr.stop
    if stop is not None:
        stop = fold_if_constant(stop)
    step = fold_if_constant(expr.step)
    slice_ = ir.Slice(start, stop, step)
    return slice_


@fold_if_constant.register
def _(expr: ir.BinOp):
    left = fold_if_constant(expr.left)
    right = fold_if_constant(expr.right)
    if left.constant and right.constant:
        oper = binops[expr.op]
        repl = oper(left, right)
    else:
        repl = ir.BinOp(left, right, op)
    return repl


@fold_if_constant.register
def _(expr: ir.UnaryOp):
    value = fold_if_constant(expr.value)
    if value.constant:
        oper = unaryops[expr.op]
        repl = wrap_constant(oper(value))
    else:
        repl = ir.UnaryOp(value, expr.op)
    return repl


@fold_if_constant.register
def _(expr: ir.Subscript):
    value = expr.value
    slice_ = fold_if_constant(expr.slice)
    repl = ir.Subscript(value, slice_)
    return repl


def discard_unbounded(iterables):
    bounded = {it for it in iterables if not (isinstance(it, ir.AffineSeq) and it.stop is None)}
    return bounded


@singledispatch
def make_affine_counter(iterable, symbols):
    msg = f"No method to make counter for {iterable}."
    raise NotImplementedError(msg)


@make_affine_counter.register
def _(iterable: ir.Subscript, symbols):
    value = iterable.value
    slice_ = fold_if_constant(iterable.slice)
    if isinstance(slice_, ir.Slice):
        if slice_.stop is None:
            stop = array_type.dims[0]
        else:
            stop = ir.Min(leading_dim, slice_.stop)
        counter = ir.AffineSeq(slice_.start, stop, slice_.step)
    else:
        # iterating over a single index subscript means iteration is bounded by
        # the second dimension.
        if symbols.lookup(value).ndims < 2:
            msg = f"Cannot iterate over a scalar reference {iterable}."
            raise ValueError(msg)
        counter = ir.AffineSeq(ir.IntNode(0), array_type.dims[1], ir.AffineSeq(1))
    return counter


@make_affine_counter.register
def _(iterable: ir.ArrayArg):
    return ir.AffineSeq(ir.IntNode(0), iterable.dims[0], ir.IntNode(1))


@make_affine_counter.register
def _(iterable: ir.NameRef):
    counter = ir.AffineSeq(ir.IntNode(0), ir.Length(iterable), ir.IntNode(1))
    return counter


@make_affine_counter.register
def _(iterable: ir.Subscript):
    if isinstance(iterable.slice, ir.Slice):
        start = iterable.start
        stop = ir.Length(iterable.value)
        step = iterable.step
        if iterable.stop is not None:
            stop = ir.Min(stop, iterable.stop)
    else:
        start = ir.IntNode(0)
        stop = ir.Length(iterable)
        step = ir.IntNode(1)
    counter = ir.AffineSeq(start, stop, step)
    return counter


def get_sequence_step(iterable):
    if isinstance(iterable, ir.Subscript):
        if isinstance(iterable.slice, ir.Slice):
            return iterable.slice.step
    elif isinstance(iterable, ir.AffineSeq):
        return iterable.step
    return ir.IntNode(1)


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

    grouped = []
    for step, counter_group in by_step.items():
        # min_interval =
        grouped.append(min_interval)


    # We can still optimize based on 1-2 uniform parameters.
    starts = set()
    stops = set()
    steps = set()

    for c in counters:
        starts.add(c.start)
        stops.add(c.stop)
        steps.add(c.step)

    if len(steps) == 1:
        step, = steps
        if len(starts) == 1:
            start, = starts
            stop = ir.Min(stops)
            counter = ir.AffineSeq(start, stop, step)
        elif len(stops) == 1:
            # Here we are better off normalizing to start at 0
            start = ir.Max(starts)
            stop, = stops
            interval_width = ir.BinOp(stop, start, "-")
            counter = ir.AffineSeq(ir.IntNode(0), interval_width, step)
        else:
            # compute minimum interval width
            pass
    else:
        # different strides, simplification steps above apply per stride,
        # with the caveat that we have to explicitly compute iteration count
        # in the end.
        pass



def make_min_sliced_len_expr(slices, leading_dim_expr):
    if not slices:
        return leading_dim_expr
    s = slices[0]
    bound = leading_dim_expr if (s.stop is None or s.start == ir.IntNode(0)) else ir.BinOp(s.stop, s.start, "-")
    for s in slices[1:]:
        b = leading_dim_expr if (s.stop is None or s.start == ir.IntNode(0)) else ir.BinOp(s.stop, s.start, "-")
        bound = ir.Ternary(ir.BinOp(bound, b, "<"), bound, b)
    return bound


def make_explicit_iter_count(counter):
    if all(isinstance(subexpr, ir.IntNode) for subexpr in counter.subexprs):
        count = numeric_max_iter_count(counter)
    elif counter.step == ir.IntNode(1):
        count = ir.BinOp(counter.stop, counter.start, "-")
    else:
        on_false = ir.BinOp(counter.stop, counter.start, "-")
        # avoid integer division
        test = ir.BinOp(interval, counter.step, "&")
        on_true = ir.BinOp(on_false, ir.IntNode(1), "+")
        count = ir.Ternary(test, on_true, on_false)
    return count



@singledispatch
def make_counter(base, syms):
    msg = f"Make counter not supported for input type {type(base)}"
    raise NotImplementedError(msg)


@make_counter.register
def _(base: ir.AffineSeq, syms):
    return base


@make_counter.register
def _(base: ir.NameRef, syms):
    if not syms.is_array(base):
        msg = f"Variable {base} is iterated over without an array type declaration or assignment."
        raise KeyError(msg)
    sym = syms.lookup(base)
    if not sym.is_array:
        msg = f"Cannot create counter from non-array type {sym.type_}"
        raise TypeError(msg)
    arr = sym.type_
    leading = arr.dims[0]
    # this is delinearized, so not a direct access func
    counter = ir.AffineSeq(ir.IntNode(0), leading, ir.IntNode(1))
    return counter


@make_counter.register
def _(base: ir.Subscript, syms):
    array_ = syms.lookup(base.value)
    # avoid linearizing
    sl = base.slice
    if isinstance(sl, ir.Slice):
        start = sl.start
        stop = arr.dims[0]
        if sl.stop is None:
            stop = sl.stop
        else:
            stop = ir.Min(stop, sl.stop)
        step = sl.step
        counter = ir.AffineSeq(start, stop, step)
    else:
        # assume single subscript
        if len(arr.dims) < 2:
            raise ValueError
        start = ir.IntNode(0)
        stop = arr.dims[1]
        stop = wrap_constant(stop)
        step = ir.IntNode(1)
        counter = ir.AffineSeq(start, stop, step)
    return counter


def make_loop_interval(targets, iterables, symbols, loop_index):
    counters = {make_counter(iterable, symbols) for iterable in iterables}
    # This has the annoying effect of dragging around the symbol table.
    # Since we need to be able to reach an array definition along each path
    # and array names are not meant to be reassigned, we should be passing an array
    # reference here.
    counter = merge_loop_counters(counters, symbols, loop_index)
    return counters
