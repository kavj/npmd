import math
import numbers
import operator
import typing
from collections import defaultdict, deque
from functools import singledispatch, singledispatchmethod

import ir
from errors import CompilerError
from utils import is_addition, is_division, is_multiplication, is_subtraction, wrap_constant, signed_integer_range
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


@singledispatchmethod
def simplify_commutative_min_max(node):
    msg = f"Internal Error: Expected min or max ir node, received {type(node)}."
    raise TypeError(msg)


@simplify_commutative_min_max.register
def _(node: ir.Max):
    numeric = set()
    seen = set()
    unchecked = deque(node.subexprs)
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
    elif numeric:
        repl = wrap_constant(max(numeric))
    else:
        # should never happen
        raise ValueError("Internal Error: commutative min max simplification.")

    return repl


@simplify_commutative_min_max.register
def _(node: ir.Min):
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
            if isinstance(value, ir.Min):
                unchecked.extend(value.values)
            else:
                unique.append(value)

    if unique:
        if numeric:
            as_const = wrap_constant(min(numeric))
            unique.append(as_const)
        if len(unique) > 1:
            repl = ir.Min(tuple(unique))
        else:
            repl, = unique
    elif numeric:
        repl = wrap_constant(min(numeric))
    else:
        # should never happen
        raise ValueError("Internal Error: commutative min max simplification.")

    return repl


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
        return expr

    @visit.register
    def _(self, expr: ir.Max):
        return expr

    @visit.register
    def _(self, expr: ir.Min):
        return expr


@singledispatch
def simplify_commutative_min_max(node):
    raise NotImplementedError


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
        return (ir.Zero, ir.Length(iterable), ir.One),

    @visit.register
    def _(self, iterable: ir.AffineSeq):
        # Note: we have to return an interval even if the stop parameter is None
        # Without this, the
        return (iterable.start, iterable.stop, iterable.step),

    @visit.register
    def _(self, iterable: ir.Subscript):
        if isinstance(iterable.slice, ir.Slice):
            # determine the inverval based on slice parameters
            # for a similar transform, see itertools.islice
            start = iterable.start
            step = iterable.step
            base = (start, ir.Length(iterable.value), step)
            if iterble.stop is None:
                intervals = base,
            else:
                intervals = base, (start, iterable.stop, step)
        else:
            intervals = (ir.Zero, ir.Length(iterable), ir.One),
        return intervals


def get_sequence_step(iterable):
    if isinstance(iterable, ir.Subscript):
        if isinstance(iterable.slice, ir.Slice):
            return iterable.slice.step
    elif isinstance(iterable, ir.AffineSeq):
        return iterable.step
    return ir.One


def consolidate_spans(spans):
    """
    Try to simplify a group of (start, stop) interval expressions.
    This does not assume that stop - start may be safely computed.
    compute stop - start.

    """

    starts = set()
    stops = set()
    for start, stop in spans:
        # we ignore anything unbounded
        # since it doesn't affect iteration count
        if stop is not None:
            starts.add(start)
            stops.add(stop)

    start_count = len(starts)
    stop_count = len(stops)

    # Check for invalid terms before proceeding.
    for term in itertools.chain(starts, stops):
        if term.constant and not isinstance(term, ir.IntConst):
            msg = f"Non-integral term {term} cannot be used as a range parameter."
            raise CompilerError(msg)

    # unbounded check
    if stop_count == 0:
        return

    if start_count == 1:
        start, = starts
        if stop_count == 1:
            stop, = stops
        else:
            stop_reduc = ir.Min(tuple(stops))
            stop = simplify_commutative_min_max(stop_reduc)
        reduced = (start, stop),

    elif stop_count == 1:
        stop, = stops
        start_reduc = ir.Max(tuple(starts))
        start = simplify_commutative_min_max(start_reduc)
        reduced = (start, stop),

    else:
        reduced = spans

    return reduced


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


class UnsafeArithmeticChecker(ExpressionVisitor):
    """
    Checks for definitely overflowing arithmetic.
    """

    def __init__(self, bitwidth):
        self.lower = -(bound + 1)
        self.upper = bound
        self.folder = const_folding()

    def overflows(self, term):
        return not (self.lower <= term.value <= self.upper)

    @singledispatchmethod
    def visit(self, expr):
        assert isinstance(expr, ir.Expression)
        return False

    @visit.register
    def _(self, expr: ir.BinOp):
        left = self.lookup(expr.left)
        right = self.lookup(expr.right)
        if left == True or right == True:
            # If the operation hasn't been folded at this point an individual
            # overflowing operand means the whole thing overflows.
            return True
        if expr.left.constant and expr.right.constant:
            # constant with neither term overflowing
            # If folding throws a compiler error, don't catch it
            # since the entire thing is unsafe rather than merely
            # overflowing at current precision.
            folded = self.folder(expr)
            if isinstance(folded, ir.IntConst):
                if self.overflows(folded):
                    return True
        return False

    @visit.register
    def _(self, expr: ir.UnaryOp):
        # don't catch compiler errors here
        folded = self.folder(expr)
        if folded.constant:
            if isinstance(folded, ir.IntConst):
                return self.overflows(folded)
        return False

    @visit.register
    def _(self, expr: ir.Constant):
        return self.overflows(expr)


def is_non_negative(expr, non_negative_terms):
    """
    check for non-negative expressions, strictly intended for finite arithmetic.

    """
    if expr.constant:
        return operator.ge(expr.value, 0)
    elif expr in non_negative_terms:
        return True
    elif isinstance(expr, ir.Max):
        return any(is_non_negative(subexpr, non_negative_terms) for subexpr in expr.subexprs)
    return False


def find_safe_interval_width(a: ir.ValueRef, b: ir.ValueRef, non_negative_terms) -> typing.Union[bool, ir.ValueRef]:
    """

    for a half-open interval [a, b)
    check if either "a" is non-negative or "b - a" is computable at compile time.
    If "b - a" overflows available fixed precision, it must be handled elsewhere.

    References:
        LIVINSKII et. al, Random Testing for C and C++ Compilers with YARPGen
        Dietz et. al, Understanding Integer Overflow in C/C++

        Bachmann et. al, Chains of Recurrences - a method to expedite the evaluation of closed-form functions
        https://gcc.gnu.org/onlinedocs/gcc/Integer-Overflow-Builtins.html
        https://developercommunity.visualstudio.com/t/please-implement-integer-overflow-detection/409051
        https://numpy.org/doc/stable/user/building.html

    """

    if a.constant:
        if b.constant:
            return wrap_constant(operator.sub(b.value, a.value))
        elif is_non_negative(b, non_negative_terms):
            return ir.BinOp(b, a, "-")
        else:
            return ir.BinOp(ir.Max(b, ir.Zero), a, "-")
    elif is_non_negative(a, non_negative_terms):
        # symbolic but strictly non-negative
        if is_non_negative(b, non_negative_terms):
            return ir.BinOp(b, a, "-")
        else:
            return ir.BinOp(ir.Max(b, ir.Zero), a, "-")
    # Todo: This should return a difference with overflow check expression.
    return


def simplify_spans(spans, non_negative_terms):
    by_stop = defaultdict(set)

    for start, stop in spans:
        by_stop[stop].add(start)

    repl = []
    seen = set()

    for _, stop in spans:
        if stop not in seen:
            starts = by_stop[stop]
            seen.add(stop)
            if len(starts) == 1:
                start, = starts
            else:
                start = ir.Max(tuple(starts))
                start = simplify_commutative_min_max(start)
            repl.append((start, stop))
    return repl


def split_intervals_by_step(intervals):
    by_step = defaultdict(set)
    for start, stop, step in intervals:
        by_step[step].add((start, stop))
    return by_step


def make_loop_interval(iterables, syms, non_negative_terms):
    """

    Make loop interval of the form (start, stop, step).

    This tries to find a safe method of calculation.
    It's probably worth forcing loops into countable range, but
    this would mean some of them require instrumentation.

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

    by_step = split_intervals_by_step(intervals)

    opt_by_step = {}

    for step, spans in by_step.items():
        opt_by_step[step] = simplify_spans(spans, non_negative_terms)

    if len(opt_by_step) == 1:
        step, spans = opt_by_step.popitem()
        if len(spans) == 1:
            span, = spans
            start, stop = span
            return ir.AffineSeq(start, stop, step)
        else:
            # produces an interval (0, diff, step)
            terms = []
            for start, stop in spans:
                diff = ir.BinOp(stop, start, "-")
                terms.append(diff)
            count = ir.Min(tuple(terms))
            interval = ir.AffineSeq(ir.Zero, count, ir.One)
            return interval
    else:
        # compute explicit interval counts
        counts = []
        seen = set()
        for step, spans in opt_by_step.items():
            diffs = []
            for start, stop in spans:
                diffs.append(ir.BinOp(stop, start, "-"))

            diff = ir.Min(tuple(diffs)) if len(diffs) != 1 else diffs.pop()
            base_count = ir.BinOp(diff, step, "//")
            test = ir.BinOp(diff, step, "%")
            fringe = ir.Ternary(test, ir.One, ir.Zero)
            count = ir.BinOp(base_count, fringe, "+")
            counts.append(count)

        count = ir.Min(tuple(counts)) if len(counts) != 1 else counts.pop()
        interval = ir.AffineSeq(ir.Zero, count, ir.One)
        return interval
