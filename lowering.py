import math
import operator
import typing
from collections import defaultdict, deque
from functools import singledispatchmethod

import ir
from errors import CompilerError
from utils import is_addition, is_multiplication, wrap_constant
from visitor import ExpressionVisitor

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

compare_ops = {"<", "<=", ">", ">=", "isnot", "in", "notin"}


def rewrite_pow(expr):
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
                raise CompilerError(msg)
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


# Todo: A lot of this can't be done consistently without type information. Some folding of constants
#       is actually useful in determining the types that can hold a constant, but they shouldn't be
#       completely folded until we have type info. Otherwise we can end up using overflowing values
#       in a way that may differ from compiler sanitization flags.


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
            return wrap_constant(value)
        else:
            # It's not possible to always place constants on the right or left due to
            # non-commutative operators, but it's okay to standardize ordering of multiplication
            # and addition with a single constant.
            if is_addition(expr):
                if left.constant:
                    return ir.BinOp(right, left, expr.op)
            elif is_multiplication(expr):
                if right.constant:
                    if not expr.in_place:
                        return ir.BinOp(right, left, expr.op)
            return ir.BinOp(left, right, expr.op)

    @visit.register
    def _(self, expr: ir.UnaryOp):
        operand = self.lookup(expr.operand)
        if operand.constant:
            # If this folds a value equivalent to MIN_INTEGER(Int64)
            # it should fail during code generation.
            op = unaryops[expr.op]
            value = op(operand.value)
            expr = wrap_constant(value)
        elif isinstance(operand, ir.UnaryOp):
            # either -(-something) or ~(~something)
            # either way, may run afoul of undefined behavior
            # if propagated
            if operand.op == expr.op:
                expr = operand.operand
            return result
        return expr

    @visit.register
    def _(self, expr: ir.Ternary):
        test = self.lookup(expr.test)
        if_expr = self.lookup(expr.if_expr)
        else_expr = self.lookup(expr.else_expr)
        if test.constant:
            return if_expr if operator.truth(test) else else_expr
        return ir.Ternary(test, if_expr, else_expr)

    @visit.register
    def _(self, expr: ir.AND):
        operands = []
        for operand in self.operands:
            operand = self.lookup(operand)
            if operand.constant:
                if not operator.truth(operand):
                    return ir.BoolConst(False)
                continue  # don't append True
            elif isinstance(operand, ir.TRUTH):
                # Truth cast is applied by AND
                operand = operand.operand
            operands.append(operand)
        num_operands = len(operands)
        if num_operands == 0:
            repl = ir.BoolConst(True)
        elif len(operands) == 1:
            # Use an explicit truth test to avoid exporting
            # "operand and True"
            operand, = operands
            repl = ir.TRUTH(operand)
        else:
            repl = ir.AND(tuple(operands))
        return repl

    @visit.register
    def _(self, expr: ir.OR):
        operands = []
        for operand in self.operands:
            operand = self.lookup(operand)
            if operand.constant:
                if operator.truth(operand):
                    return ir.BoolConst(True)
                continue  # don't append constant False
            elif isinstance(operand, ir.TRUTH):
                # Truth cast is applied by OR
                operand = operand.operand
            operands.append(operand)
        num_operands = len(operands)
        if num_operands == 0:
            return ir.BoolConst(False)
        elif num_operands == 1:
            # Use an explicit truth test to avoid exporting
            # "expr or False"
            operand, = operands
            repl = ir.TRUTH(operand)
        else:
            repl = ir.OR(tuple(operands))
        return repl

    @visit.register
    def _(self, expr: ir.NOT):
        operand = self.lookup(expr.operand)
        if isinstance(operand, ir.Constant):
            value = not operator.truth(operand.value)
            expr = wrap_constant(value)
        elif isinstance(operand, ir.TRUTH):
            # NOT implicitly truth tests, so
            # we can discard the explicit test.
            expr = ir.NOT(operand.operand)
        return operand

    @visit.register
    def _(self, expr: ir.XOR):
        # not sure about this one, since
        # it's not monotonic
        raise NotImplementedError

    @visit.register
    def _(self, expr: ir.Max):
        raise NotImplementedErrorf

    @visit.register
    def _(self, expr: ir.Min):
        raise NotImplementedError


def simplify_commutative_max(node):
    numeric = set()
    seen = set()
    unchecked = deque(node.values)
    unique = deque()

    while unchecked:
        value = unchecked.popleft()
        if value.constant:
            numeric.add(value.value)
        elif value not in seen:
            seen.add(value)
            if isinstance(value, ir.Max):
                # Since this handles expressions that are assumed to
                # be commutative and not overflow or contain unordered
                # operands, we can inline nested max terms the first time
                # a unique max expression is encountered.
                unchecked.extend(value.values)
            else:
                unique.append(value)

    if unique:
        if numeric:
            as_const = wrap_constant(max(numeric))
            unique.append(as_const)
        if len(unique) > 1:
            repl = ir.Max(tuple(unique))
        else:
            repl, = unique
    else:
        repl = numeric

    return repl


def unwrap_truth_tested(expr):
    """
    Extract truth tested operands. This helps limit isinstance checks for cases
    where an enclosing expression will refer to the truth test of expr, with or
    without an explicit TRUTH node wrapper.
    """
    if isinstance(expr, ir.TRUTH):
        expr = expr.operand
    return expr


class arithmetic_folding(ExpressionVisitor):
    """
    Non-recursive visitor for folding identity ops

    This does not respect unordered operands such as nans. It's primarily intended for integer
    arithmetic.

    """

    def __call__(self, expr):
        return self.lookup(expr)

    @singledispatchmethod
    def visit(self, expr):
        raise NotImplementedError

    @visit.register
    def _(self, node: ir.BinOp):
        left = node.left
        right = node.right
        # if a constant expression shows up here, treat it as an error since
        # it's weirder to handle than it seems
        assert not (left.constant and right.constant)
        two = ir.IntConst(2)
        negative_one = ir.IntConst(-1)

        if is_pow(node):
            if right == ir.Zero:
                return ir.One
            elif right == ir.One:
                return left
            elif right == two:
                return ir.BinOp(left, left, "*=" if node.in_place else "*")
            # square roots shouldn't come up here, given the associative qualifier
        elif is_addition(node):
            if right == ir.Zero:
                return left
            elif equals_unary_negate(right):
                return ir.BinOp(left, right.operand, "-=" if node.in_place else "-")
            elif equals_unary_negate(left):
                assert not node.in_place
                return ir.BinOp(right, left.operand, "-")
        elif is_subtraction(node):
            if left == ir.Zero:
                return ir.UnaryOp(right, "-")
            elif right == ir.Zero:
                return left
            elif equals_unary_negate(right):
                return ir.BinOp(left, right.operand, "+=" if node.in_place else "+")
            elif equals_unary_negate(left):
                assert not node.in_place
                return ir.BinOp(right, left.operand, "+")
        elif is_division(node):
            if right == ir.Zero:
                msg = f"Divide by zero error in expression {node}."
                raise CompilerError(msg)
            elif node.op in ("//", "//="):
                # only safe to fold floor divide, ignore left == right since these might
                # be zero. Constant cases should be handled by the const folder.
                if left == ir.Zero or right == ir.One:
                    return left
        elif is_multiplication(node):
            if left == ir.Zero:
                return ir.Zero
            elif left == ir.One:
                return right
            elif left == negative_one:
                if equals_unary_negate(right):
                    # -(-something)) is safe in Python but possibly unsafe in a fixed width
                    # destination. Folding it should be considered safe.
                    return right.operand
                else:
                    return ir.UnaryOp(right, "-")

    @visit.register
    def _(self, node: ir.UnaryOp):
        if isinstance(node.operand, ir.UnaryOp) and node.operand == node.op:
            return node.operand.operand
        else:
            return node

    @visit.register
    def _(self, node: ir.AND):
        seen = set()
        operands = []
        for operand in node.operands:
            # constant folding should be separate
            # as it causes too many issues here
            assert not operand.constant
            operand = unwrap_truth_tested(operand)
            if operand not in seen:
                seen.add(operand)
                operands.append(operand)
        if len(operands) > 1:
            return ir.AND(tuple(operands))
        else:
            operand, = operands
            return ir.TRUTH(operand)

    @visit.register
    def _(self, node: ir.OR):
        seen = set()
        operands = []
        for operand in node.operands:
            assert not operand.constant
            operand = unwrap_truth_tested(operand)
            if operand not in seen:
                operands.append(operand)
        if len(operands) > 1:
            return ir.OR(tuple(operands))
        else:
            operand, = operands
            return ir.TRUTH(operand)

    @visit.register
    def _(self, node: ir.NOT):
        if not applies_truth_test(node.operand):
            return node
        if isinstance(node.operand, ir.BinOp):
            op = node.operand.op
            if op == "==":
                left = node.operand.left
                right = node.operand.right
                return ir.BinOp(left, right, "!=")
            elif op == "!=":
                left = node.operand.left
                right = node.operand.right
                return ir.BinOp(left, right, "==")
            # >, >=, <, <= are not safe to invert if unordered operands
            # are present, particularly floating point NaNs.
            # While this started off assuming integer arithmetic, it may
            # be better to move this after typing, since some of this applies
            # equally or almost as well to floating point arithmetic.
        elif isinstance(node, ir.NOT):
            # remove double negation
            operand = node.operand.operand
            if not applies_truth_test(operand):
                # If the unwrapped type doesn't already export a truth test
                # we need to indicate this explicitly.
                operand = ir.TRUTH(operand)
            return operand
        return node

    @visit.register
    def _(self, node: ir.TRUTH):
        # This will leave truth casts on constant integers
        # and floats, since the only gain there is a loss
        # of clarity.
        if applies_truth_test(node.operand):
            node = node.operand
        return node

    @visit.register
    def _(node: ir.Ternary):
        if node.if_expr == node.else_expr:
            return if_expr
        elif node.test.constant:
            return node.if_expr if operator.truth(node.test) else node.else_expr
        test = node.test
        if isinstance(test, ir.BinOp):
            if_expr = node.if_expr
            else_expr = node.else_expr
            test = node.test
            if test.op in ("<", "<="):
                if if_expr == test.left and else_expr == test.right:
                    return ir.Min((if_expr, else_expr))
                elif else_expr == test.right and if_expr == test.left:
                    # This is almost negated. The issue is if in the destination assembly:
                    #
                    #     min(a,b) is implemented as a if a <= b else b
                    #     max(a,b) is implemented as a if a >= b else b
                    #
                    #  which is common, we reverse operand order to properly catch unordered cases
                    #  This does not follow Python's min/max conventions, which are too error prone.
                    #  Those can arbitrarily propagate or suppress nans as a side effect of
                    #  determining type from the leading operand.
                    return ir.Max((else_expr, if_expr))

            elif test.op in (">", ">="):
                if if_expr == test.left and else_expr == test.right:
                    return ir.Max((if_expr, else_expr))
                elif if_expr == test.right and else_expr == test.left:
                    # right if left < right else left
                    return ir.Min((else_expr, if_expr))
        return node


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
            intervals = {(ir.Zero, ir.Length(iterable), ir.One), }
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
    value = -(2 ** p)
    return ir.IntConst(value)


def MAX_INT(p: int) -> ir.IntConst:
    """
    max integer with p bits
    """

    assert p > 0
    value = 2 ** p - 1
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


def sub_is_safe(a: ir.ValueRef, b: ir.ValueRef, p: int) -> typing.Union[bool, ir.ValueRef]:
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

    # Todo: This should probably avoid doing explicit overflow checks when they cannot be fully resolved.
    #  Rather we can consolidate any arithmetic that might overflow, then use compiler specific extensions
    #  for any remaining checks.
    assert p > 0
    imin = MIN_INT(p)
    imax = MAX_INT(p)

    # Check for simple case, ignoring intermediate overflow
    expr = fold_constants(ir.BinOp(a, b, "-"))

    if expr.constant:
        truth_value = imin.value <= expr.value <= imax.value
        return ir.BoolConst(truth_value)
    elif b == ir.Zero:
        # It's always safe to compute a - 0
        # It's unsafe to compute 0 - MIN_INT(p)
        return ir.BoolConst(True)
    else:
        # Check for
        # b >= 0 and (a >= 0 or INT_MIN(p) + b <= a)
        # b =< 0 and (a <= 0 or a <= INT_MAX(p) + b)

        b_ge_zero = ir.BinOp(b, ir.Zero, ">=")
        imin = MIN_INT(p)
        imax = MAX_INT(p)

        fold_constants = const_folding()

        a_ge_zero = fold_constants(ir.BinOp(a, ir.Zero, ">="))
        a_le_zero = fold_constants(ir.BinOp(a, ir.Zero, "<="))

        b_ge_zero = fold_constants(ir.BinOp(b, ir.Zero, ">="))

        b_plus_min = fold_constants(ir.BinOp(b, imin, "+"))
        b_plus_max = fold_constants(ir.BinOp(b, imax, "+"))

        b_plus_min_le_a = fold_constants(ir.BinOp(b_plus_min, a, "<="))
        a_le_b_plus_max = fold_constants(ir.BinOp(a, b_plus_max, "<="))

        on_true = ir.OR((a_ge_zero, b_plus_min_le_a))
        on_false = ir.OR((a_le_zero, a_le_b_plus_max))
        test = ir.Ternary(b_ge_zero, )
        on_true = ir.OR(a_ge_zero, on_true, on_false)

        test_on_true = ir.OR((ir.BinOp(a, ir.Zero, ">="), ir.BinOp(ir.BinOp(b, imin, "+"), a, "<=")))
        test_on_false = ir.OR((ir.BinOp(a, ir.Zero, "<="), ir.BinOp(a, ir.BinOp(b, imax, "+"), "<=")))
        test = ir.Ternary(non_negative_b, test_on_pos, test_on_non_neg)
        return ir.TRUTH(test)


def split_intervals_by_step(intervals):
    pass


def split_start_stop(intervals):
    """
    split pairs of start stop into two dictionaries,

    starts with respect to stops
    stops with respect to starts

    """

    by_start = defaultdict(set)
    by_stop = defaultdict(set)

    for start, stop in intervals:
        by_start[start].add(stop)
        by_stop[stop].add(start)

    return by_start, by_stop


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
