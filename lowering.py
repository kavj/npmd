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


def unpack_assignment(target, value, pos):
    if isinstance(target, ir.Tuple) and isinstance(value, ir.Tuple):
        if target.length != value.length:
            msg = f"Cannot unpack {value} with {value.length} elements using {target} with {target.length} elements: " \
                  f"line {pos.line_begin}."
            raise ValueError(msg)
        for t, v in zip(target.subexprs, value.subexprs):
            yield from unpack_assignment(t, v, pos)
    else:
        yield target, value


def unpack_iterated(target, iterable, pos):
    if isinstance(iterable, ir.Zip):
        # must unpack
        if isinstance(target, ir.Tuple):
            if len(target.elements) == len(iterable.elements):
                for t, v in zip(target.elements, iterable.elements):
                    yield from unpack_iterated(t, v, pos)
            else:
                msg = f"Mismatched unpacking counts for {target} and {iterable}, {len(target.elements)} " \
                      f"and {(len(iterable.elements))}."
                raise ValueError(msg)
        else:
            msg = f"Zip construct {iterable} requires a tuple for unpacking."
            raise ValueError(msg)

    else:
        # Array or sequence reference, with a single opaque target.
        yield target, iterable


def is_pow(expr):
    return isinstance(expr, ir.BinOp) and expr.op in ("**", "**=")


def is_fma_pattern(expr):
    """
    This ignores safety issues, which may be addressed later for anything
    that looks like a * b - c * d.


    """

    if isinstance(expr, ir.BinOp) and expr.op in ("+", "+=", "-", "-="):
        left = expr.left
        right = expr.right
        for operand in (left, right):
            if isinstance(operand, ir.BinOp):
                if operand.op == "*":
                    return True
            elif isinstance(operand, ir.UnaryOp):
                # Expression simplifiers should have already folded any multiple
                # nestings of unary -
                if (operand.op == "-"
                        and isinstance(operand.operand, ir.BinOp)
                        and operand.operand.op == "*"):
                    return True
    return False


def rewrite_pow(expr):
    if not is_pow(expr):
        return expr
    coeff = expr.right
    base = expr.left
    if coeff == ir.Zero:
        return ir.One
    elif base == ir.Zero:
        # checking for weird errors more than anything
        if coeff.constant:
            if operator.lt(coeff.value, 0):
                # this isn't intended to catch all edge cases, just an obvious
                # one that may come up after folding
                msg = f"raises 0 to a negative power {expr}."
                raise ValueError(msg)
            else:
                return ir.Zero
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


class flatten_exprs(ExpressionVisitor):

    def __init__(self, syms, min_max_is_ordered=True):
        self._min_max_is_ordered = min_max_is_ordered
        self.syms = syms

    def __call__(self, expr):
        return self.lookup(expr)

    # remove nesting from min/max where possible
    # de-nest boolean expressions


class simplify_exprs(ExpressionVisitor):
    """
    Fold redundancy, typically stuff that is added by another pass, since it's not necessarily
    tractable to catch every case of expression blowup where it occurs.

    References

    LIVINSKII et. al, Random Testing for C and C++ Compilers with YARPGen


    """

    def __init__(self, syms, min_max_is_ordered=True):
        self._min_max_is_ordered = min_max_is_ordered
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
            # sanitize some cases of potential intermediate overflow
            elif isinstance(right, ir.UnaryOp):
                if right.op == "-":
                    if op == "+":
                        return ir.BinOp(left, right.operand, "-")
                    elif op == "-":
                        return ir.BinOp(left, right.operand, "+")
                    elif op == "+=":
                        return ir.BinOp(left, right.operand, "-=")
                    elif op == "-=":
                        return ir.BinOp(left, right.operand, "+=")
            elif isinstance(left, ir.UnaryOp):
                # lower priority, since it inverts operand ordering
                if left.op == "-":
                    if op == "+":
                        return ir.BinOp(right, left.operand, "-")
                    elif op == "-":
                        return ir.BinOp(right, left.operand, "-")
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
        test = self.lookup(expr.test)
        if_expr = self.lookup(expr.if_expr)
        else_expr = self.lookup(expr.else_expr)
        if isinstance(test, ir.BinOp):
            # Todo: This needs a flag of some sort, since this only applies
            #       if min and max semantics are of the common non-IEEE variety
            if self._min_max_is_ordered:
                if test.op in ("<", "<="):
                    if if_expr == test.left and else_expr == test.right:
                        return ir.Min((if_expr, else_expr))
                elif test.op in (">", ">="):
                    if if_expr == test.left and else_expr == test.right:
                        return ir.Max((if_expr, else_expr))
        return expr

    @visit.register
    def _(self, expr: ir.AND):
        seen = set()
        repl = []
        for operand in expr.operands:
            operand = self.lookup(operand)
            if operand not in seen:
                seen.add(operand)
                repl.append(operand)
        return ir.AND(tuple(repl))

    @visit.register
    def _(self, expr: ir.OR):
        seen = set()
        repl = []
        for operand in expr.operands:
            operand = self.lookup(operand)
            if operand not in seen:
                seen.add(operand)
                repl.append(operand)
        return ir.OR(tuple(repl))

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


class const_folding(ExpressionVisitor):
    # These don't reject large integer values, since they may be intermediate values.
    # It only matters if they are explicitly bound to a fixed width data type that
    # cannot contain them.

    def __call__(self, expr):
        return self.lookup(expr)

    @singledispatchmethod
    def visit(self, expr):
        return super().visit(expr)

    @visit.register
    def _(self, expr: ir.BinOp):
        left = self.lookup(expr.left)
        right = self.lookup(expr.right)
        if left.constant and right.constant:
            op = binops[expr.op]
            if op in ("<<", ">>", "<<=", ">>="):
                 if not isinstance(right, ir.IntConst):
                     msg = f"Cannot safely evaluate shifts by non-integral amounts: {left.value}  {op} {right.value}."
                     raise ValueError(msg)
                 elif operator.eq(right.value, 0):
                     msg = f"Shift by zero error: {left.value} {op} {right.value}"
                     raise ValueError(msg)
                 elif operator.lt(right.value, 0):
                     msg = f"Shift amount cannot be negative: {left.value} {op} {right.value}"
                     raise ValueError(msg)
            value = op(left.value, right.value)
            result = wrap_constant(value)
            return result
        else:
            return expr

    @visit.register
    def _(self, expr: ir.UnaryOp):
        operand = self.lookup(expr.operand)
        if operand.constant:
            # If this folds a value equivalent to MIN_INTEGER(Int64)
            # it should fail during code generation.
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
        if len(operands) > 1:
            repl = ir.OR(tuple(operands))
        else:
            operand = operands.pop()
            repl = ir.TRUTH(operand)
        return repl

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

    def __call__(self, expr):
        repl = self.lookup(expr)
        return repl

    @singledispatchmethod
    def visit(self, iterable):
        msg = f"No method to make counter for {iterable}."
        raise NotImplementedError(msg)

    @visit.register
    def _(self, iterable: ir.NameRef):
        return {(ir.Zero, ir.Length(iterable), ir.One)}

    @visit.register
    def _(self, iterable: ir.AffineSeq):
        # Note: we have to return an interval even if the stop parameter is None
        # Without this, the
        return {(iterable.start, iterable.stop, iterable.step)}

    @visit.register
    def _(self, iterable: ir.Subscript):
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


def find_min_interval_width(spans):
    starts = set()
    stops = set()
    for start, stop in spans:
        # we ignore anything unbounded
        # since it doesn't affect iteration count
        if stop is not None:
            starts.add(start)
            stops.add(stop)

    stop_count = len(stops)
    if stop_count == 0:
        # unbounded
        return

    single_start = len(starts) == 1
    single_stop = stop_count == 1

    if single_start or single_stop:
        if single_start and single_stop:
            start, = starts
            stop, = stops
        elif single_start:
            start, = starts
            stop = ir.Min(tuple(stops))
        else:
            start = ir.Max(tuple(starts))
            stop, = stops
        min_width = ir.BinOp(stop, start, "-")
    else:
        widths = set()
        for start, stop in start_stop:
            if start == ir.Zero:
                widths.add(stop)
            elif stop is not None:
                d = ir.BinOp(stop, start, "-")
                widths.add(d)
        min_width = ir.Min(tuple(widths))
    return width


def unary_minus_is_safe(a: ir.ValueRef, p: int) -> ir.ValueRef:
    assert p >= 0
    min_int = -1 * operator.pow(2, p - 1)
    if a.constant:
        cond = operator.ne(a, min_int)
    else:
        cond = ir.BinOp(a, ir.IntConst(min_int), "!=")
    return cond

# Stubs for range checks, not always decidable
# The expressions themselves should be evaluated as if safe and well defined.

def is_non_negative(expr):
    return NotImplemented


def is_positive(expr):
    return NotImplemented


def is_negative(expr):
    return NotImplemented


def MIN_INT(p: int) -> ir.IntConst:
    """
    minimum integer with p bits
    """

    assert p > 0
    value = -(2**p)
    return ir.IntConst(value)


def MAX_INT(p: int) -> ir.IntConst:
    """
    max integer with p bits
    """

    assert p > 0
    value = 2**p - 1
    return ir.IntConst(value)


def add_is_safe(a: ir.ValueRef, b: ir.ValueRef, p: int) -> ir.ValueRef:
    """
    This is meant to test whether we can safely compute an expression at runtime, particularly
    expressions that are generated during compile time lowering.

    a + b can be safely evaluated at runtime if:
    
    Note that folding at compile time uses Python's arbitrary precision integers
    
    "a + b" can be safely evaluated at runtime if 
    
    INT_MIN(p) <= a + b <= INT_MAX(p)
    
    which holds if any of the following conditions hold. 
    
    Note the >= and <= are partially redundant with the first condition but still valid.
    
    (1) a == 0 or b == 0
    
    (2) a >= 0 and b <= MAX_INT(p) - a
    
    (3) a <= 0 and MIN_INT(p) - a <= b
    
    (4) b => 0 and a <= MAX_INT(p) - b
    
    (5) b <= 0 and MIN_INT(p) - b <= a
    
    References:
        LIVINSKII et. al, Random Testing for C and C++ Compilers with YARPGen
        Dietz et. al, Understanding Integer Overflow in C/C++
        Bachmann et. al, Chains of Recurrences - a method to expedite the evaluation of closed-form functions
        https://gcc.gnu.org/onlinedocs/gcc/Integer-Overflow-Builtins.html
        https://developercommunity.visualstudio.com/t/please-implement-integer-overflow-detection/409051
        https://numpy.org/doc/stable/user/building.html
    """
    assert p > 0
    mag = operator.pow(2, p)
    imin = MIN_INT(p)
    imax = MAX_INT(p)
    if a == ir.Zero or b == ir.Zero:
        # a == 0 or b == 0
        return ir.BoolConst(True)
    elif a.constant and b.constant:
        # safely computable at compile time using arbitrary precision integers
        res = operator.sub(a.value, b.value)
        cond = (MIN_INT(p).value <= res <= MAX_INT(p).value)
        return ir.BoolConst(cond)
    # Todo: This can be optimized quite a bit if we can determine whether values are positive or negative here.
    elif a.constant:
        if operator.gt(a.value, imax.value):
            # overflowing constant
            return ir.BoolConst(False)
        elif operator.gt(a.value, 0):
            # a > 0 and b <= MAX_INT(p) - a
            diff = ir.BinOp(imin.value, a, "-")
            return ir.BinOp(b, diff, "<=")
        else:
            # a < 0 and MIN_INT(p) - a <= b
            diff = ir.BinOp(imin.value, a, "-")
            return ir.BinOp(diff, b, "<=")
    elif b.constant:
        if operator.gt(b.value, imax.value):
            # overflowing constant
            return ir.BoolConst(False)
        if operator.gt(b.value, 0):
            # b > 0 and a <= MAX_INT(p) - b
            diff = ir.BinOp(imax.value, b, "-")
            return ir.BinOp(a, diff, "<=")
        else:
            # b < 0 and MIN_INT(p) - b <= a
            diff = ir.BinOp(imin.value, b, "-")
            return ir.BinOp(diff, a, "<=")
    else:
        non_negative_a = ir.BinOp(a, ir.Zero, ">=")
        cond_if_true = ir.BinOp(b, ir.BinOp(imax, a, "-"))
        cond_if_false = ir.BinOp(ir.BinOp(imin, a, "-"), b, "<=")
        return ir.Ternary(non_negative_a, cond_if_true, cond_if_false)


def sub_is_safe(a: ir.ValueRef, b: ir.ValueRef, p: int)-> typing.Union[bool, ir.ValueRef]:
    """

    Note that folding at compile time uses Python's arbitrary precision integers

    "a - b" can be safely evaluated at runtime if:

        (b == 0) or (INT_MIN(p) <= a - b <= INT_MAX(p))

        which implies a - b is safe if any of the following lines hold:
            b == 0
            b < 0 and (a < 0 or a <= INT_MAX(p) + b)
            b > 0 and (a > 0 or INT_MIN(p) + b <= a)

    This should convert to an OverflowError if something cannot be safely evaluated.

    References:
        LIVINSKII et. al, Random Testing for C and C++ Compilers with YARPGen
        Dietz et. al, Understanding Integer Overflow in C/C++
        Bachmann et. al, Chains of Recurrences - a method to expedite the evaluation of closed-form functions
        https://gcc.gnu.org/onlinedocs/gcc/Integer-Overflow-Builtins.html
        https://developercommunity.visualstudio.com/t/please-implement-integer-overflow-detection/409051
        https://numpy.org/doc/stable/user/building.html

    """
    assert p > 0

    if a.constant and b.constant:
        c = operator.sub(a.value, b.value)
        k = operator.pow(2, p-1)
        return -k <= c < k
    elif b == ir.Zero:
        # Note this is not safe if a == 0, since b == INT_MIN(p) would overflow.
        # This is always safe and may be optimized away somewhere else.
        return ir.BoolConst(True)
    else:
        # I'll probably need to change the signature here later, since this really needs
        # dataflow information, which isn't really well established yet.
        raise NotImplementedError
    return ir.BoolConst(False)


def make_loop_counter(iterables, syms):
    """

    Map a combination of array iterators and range and enumerate calls to a set of counters,
    which capture the appropriate intervals.

    This carries the requirement that all parameters have non-negative values and that step parameters
    are strictly positive. Without that, it would take too much arithmetic to sanitize all cases at runtime
    in a way that doesn't rely on things such as non-portable saturating arithmetic routines.

    For example, suppose we have

    for u, v in zip(array0[i::k], array1[j::k]):
        ...

    where its known at compile time that n == len(array0) == len(array1)

    This means we have intervals

    (i, n, k), (j, n, k)

    which results in a loop counter with parameters:

    (max(i, j), n, k)

    Ignoring the issue of sanitizing the loop step there against overflow, we need an exact iteration count
    so that the corresponding lowering looks like

    array0[i + loop_index * k]
    array1[j + loop_index * k]

    which is unsafe if we cannot safely compute n - max(i, j) at runtime

    If all bound variable are non-negative, we can always safely compute this stuff, and
    sanitizing a non-unit step can be done by computing:
        stop = max(start, stop)
        stop -= (stop - start) % step

    Without this, we have greater complexity in order to avoid injecting unsafe behavior during lowering
    that was safe in the original source.


    References:
        LIVINSKII et. al, Random Testing for C and C++ Compilers with YARPGen
        Dietz et. al, Understanding Integer Overflow in C/C++
        Bachmann et. al, Chains of Recurrences - a method to expedite the evaluation of closed-form functions
        https://gcc.gnu.org/onlinedocs/gcc/Integer-Overflow-Builtins.html
        https://developercommunity.visualstudio.com/t/please-implement-integer-overflow-detection/409051
        https://numpy.org/doc/stable/user/building.html

    """

    # Todo: This should be a loop transform pass, since it needs to inject error handling for complicated parameters
    #       and eventually handle reversal.

    intervals = set()
    split_intervals = interval_splitting()

    for iterable in iterables:
        from_iterable = split_intervals(iterable)
        intervals.update(from_iterable)

    unique_starts = set()
    unique_stops = set()
    unique_steps = set()

    for start, stop, step in intervals:
        unique_starts.add(start)
        unique_stops.add(stop)
        unique_steps.add(step)

    unique_stops.discard(None)

    # This should never fail.
    assert unique_stops

    have_unique_start = len(unique_starts) == 1
    have_unique_stop = len(unique_stops) == 1
    have_unique_step = len(unique_steps) == 1

    if have_unique_step:
        step, = unique_steps
        if have_unique_start:
            # simple case
            start, = unique_starts
            if have_unique_stop:
                stop, = unique_stops
            else:
                stop = ir.Min(tuple(unique_stops))
        elif have_unique_stop:
            # Todo: Even with step size > 0, we can't compute stop - start here unless we can prove
            #       INT_MIN <= stop - start <= INT_MAX. It's probably worth checking this, since start
            #       should virtually always be >= 0 in the absence of coding mistakes. Otherwise it pessimizes
            #       code generation quite a bit. We could always enforce this.
            start = ir.Max(tuple(unique_starts))
            stop = ir.BinOp(stop, start, "-")
        else:
            # Normalize bounds after reducing over all bounds
            # but still fold step. This is unsafe if parameters
            # may be negative.
            widths = set()
            for start, stop, _ in intervals:
                safe_stop = ir.Max(start, stop)
                widths.add(ir.BinOp(safe_stop, start, "-"))
            start = ir.Zero
            stop = ir.Min(tuple(widths))
            # Todo: This needs a sanitizer check for MAX_INT - stop < step
        counter = ir.AffineSeq(start, stop, step)
    else:
        by_step = defaultdict(set)
        counts = set()

        for start, stop, step in intervals:
            by_step[step].add((start, stop))

        for step, starts_stops in by_step.items():
            # again, safe with
            min_width = find_min_interval_width(starts_stops)
            if step == ir.One:
                count = min_width
            else:
                base_count = ir.BinOp(min_width, step, "//")
                fringe_test = ir.BinOp(min_width, step, "%")
                fringe_count = ir.Ternary(fringe_test, ir.One, ir.Zero)
                count = ir.BinOp(base_count, fringe_count, "+")
            counts.add(count)
        counter = ir.AffineSeq(ir.Zero, ir.Min(tuple(counts)), ir.One)
    return counter
