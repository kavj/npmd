import operator
import typing
from collections import defaultdict, deque
from functools import singledispatch, singledispatchmethod

import ir
from errors import CompilerError
from utils import is_addition, is_division, is_multiplication, is_pow, is_subtraction, is_truth_test, wrap_constant, \
    unpack_iterated, equals_unary_negate
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
        return expr

    @visit.register
    def _(self, expr: ir.Select):
        test = self.lookup(expr.test)
        on_true = self.lookup(expr.on_true)
        on_false = self.lookup(expr.on_false)
        if test.constant:
            return on_true if operator.truth(test) else on_false
        return ir.Select(test, on_true, on_false)

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
                # Todo: this is not entirely correct... as it may not be a unary node
                # need something like extract unary operand..
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
        if not is_truth_test(node.operand):
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
            if not is_truth_test(operand):
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
        if is_truth_test(node.operand):
            node = node.operand
        return node

    @visit.register
    def _(node: ir.Select):
        if node.on_true == node.on_false:
            return node.on_true
        elif node.test.constant:
            return node.on_true if operator.truth(node.test) else node.on_false
        test = node.test
        if isinstance(test, ir.BinOp):
            on_true = node.on_true
            on_false = node.on_false
            test = node.test
            if test.op in ("<", "<="):
                if on_true == test.left and on_false == test.right:
                    return ir.Min((on_true, on_false))
                elif on_false == test.right and on_true == test.left:
                    # This is almost negated. The issue is if in the destination assembly:
                    #
                    #     min(a,b) is implemented as a if a <= b else b
                    #     max(a,b) is implemented as a if a >= b else b
                    #
                    #  which is common, we reverse operand order to properly catch unordered cases
                    #  This does not follow Python's min/max conventions, which are too error prone.
                    #  Those can arbitrarily propagate or suppress nans as a side effect of
                    #  determining type from the leading operand.
                    return ir.Max((on_false, on_true))

            elif test.op in (">", ">="):
                if on_true == test.left and on_false == test.right:
                    return ir.Max((on_true, on_false))
                elif on_true == test.right and on_false == test.left:
                    # right if left < right else left
                    return ir.Min((on_false, on_true))
        return node


@singledispatch
def interval_from_iterable(iterable):
    msg = f"Cannot retrieve interval from iterable of type {iterable}."
    raise NotImplementedError(msg)


@interval_from_iterable.register
def _(iterable: ir.AffineSeq):
    return (iterable.start, iterable.stop, iterable.step)


@interval_from_iterable.register
def _(iterable: ir.NameRef):
    return (ir.Zero, ir.Length(iterable), ir.One)


@interval_from_iterable.register
def _(iterable: ir.Subscript):
    if isinstance(iterable.slice, ir.Slice):
        slice_ = iterable.slice
        start = slice_.start
        if start is None:
            start = ir.Zero
        stop = ir.SingleDimRef(iterable.target, ir.Zero)
        if slice_.stop is not None:
            stop = ir.Min(stop, slice_.stop)
        step = slice_.step
        if step is None:
            step = ir.One
    else:
        start = ir.Zero
        stop = ir.SingleDimRef(iterable, ir.One)
        step = ir.One
    return start, stop, step


def try_compute_simple_loop_interval(header: ir.ForLoop):

    starts = defaultdict(set)
    stops = defaultdict(set)

    for _, iterable in unpack_iterated(header):
        start, stop, step = interval_from_iterable(iterable)
        starts.add(start)
        stops.add(stop)
        steps.add(step)
        if len(steps) > 1:
            # no foldable option
            return

    step = steps.pop()

    if len(starts) == 1:
        if len(stops) == 1:
            start = starts.pop()
            stop = stops.pop()
            # simple interval
        else:
            stop  = ir.Min(tuple(stops))
    elif len(stops) == 1:
        start = ir.Max(tuple(starts))
        stop = stops.pop()
    else:
        # no foldable option
        return
    return start, stop, step


def make_count_expr(diff, step):
    if step == ir.Zero:
        msg = "Zero step loop iterator encountered."
        raise CompilerError(msg)
    elif step == ir.One:
        return diff
    on_false = ir.BinOp(diff, step, "//")
    modulo = ir.BinOp(diff, step, "%")
    on_true = ir.BinOp(on_false, ir.One, "+")
    return ir.Select(predicate=modulo, on_true=on_true, on_false=on_false)


def reduce_per_step_diff_sets(by_step):
    reduced = defaultdict(set)
    # we need sets rather than tuples here to avoid
    for step, diffs in by_step.items():
        diffs = frozenset(diffs)
        reduced[diffs].add(step)
    remerged = {}
    for diffs, steps in reduced.items():
        max_step = ir.Max(frozenset(steps))
        if max_step in remerged:
            raise CompilerError("collision... finish this part")
        remerged[max_step] = diffs
    if len(remerged) < len(by_step):
        return remerged
    return by_step


def make_explicit_count(header: ir.ForLoop):
    # This does not use safe subtraction
    simplify_diff = arithmetic_folding()
    by_step = defaultdict(set)
    diffs = set() # diffs are cacheable
    for _, iterable in unpack_iterated(header):
        start, stop, step = interval_from_iterable(iterable)
        d = ir.BinOp(stop, start, "-")
        # remove trivial arithmetic
        d = simplify_diff(d)
        diffs.add(d)
        by_step[step].add(d)

    # Since division is expensive, compute exact step count
    # only on the minimum difference per step.

    # Find commmon components
    all_diff_sets = tuple(frozenset(v) for v in by_step.values())
    common = set.intersection(*all_diff_sets)

    # factor out shared component
    if len(common) > 0:
        for diffs in by_step.values():
            diffs.difference_update(shared)
        # If the common component equals the diff set
        # for any sets, then we can consider only the max
        # step for these
        step_reduc = set()
        for step, diffs in by_step.values():
            if not diffs:
                step_reduc.add(step)
        for step in step_reduc:
            by_step.pop(step)
        max_common = ir.Max(tuple(step_reduc))
        by_step[max_common].update(common)







def make_openmp_compatible_loop(header: ir.ForLoop, symbols):
    """
    Make single index loop without break.
    This is unsafe if it's unsafe to explicitly compute trip count. It probably needs overflow checks.

    """

    by_start_step = defaultdict(set)
    by_step = defaultdict(set)
    by_iterable = {}
    for _, iterable in unpack_iterated(header):
        start, stop, step = interval_from_iterable(iterable)
        key = (start, step)
        by_start_step[key].add(stop)
        by_step[step].add((start, stop))
        by_iterable[iterable] = key

    # If this uses a single step, and it either uses a single start
    # or a single stop, then use a simpler folding method.
    if len(by_step) == 1:
        step, pairs = by_step.popitem()
        starts = set()
        stops = set()
        for start, stop in pairs:
            starts.add(start)
            stops.add(stop)
        if len(starts) == 1:
            pass

    by_start_step_opt = {}
    loop_params = {}

    # make index variables

    for key, stop in by_start_step.items():
        stop = ir.Min(stop)
        stop = simplify_commutative_min_max(stop)
        index_var = symbols.register_unique_name(prefix="i")
        loop_params[key] = stop, iv

    # Getting well defined explicit iteration counts has a lot of edge cases, including
    # division by zero and possible arithmetic overflow, where checks for arithmetic overflow
    # are compiler specific. In addition, there's the cost of potentially computing division for low trip counts
    # on multiple step sizes. It's easier just to take the naive approach and put all additional tests at the beginning
    # of the loop body.

    lp = iter(loop_params.items())
    loop_index_params = next(lp)

    initializers = []

    # anything other than the primary loop index is tested at the end of the loop body
    # This doesn't preserve initial testing order.

    # epilogue = []
    repl_body = []
    pos = header.pos
    # now set up the initial assignments

    for target, iterable in unpack_iterated(header):
        key = by_iterable[iterable]
        _, iv = loop_params[key]
        if isinstance(iterable, ir.AffineSeq):
            value = iv
        else:
            value = ir.Subscript(iterable, iv)
        assign = ir.Assign(target, value)
        repl_body.append(assign)

    repl_body.extend(header.body)

    for (start, step), (stop, iv) in lp:
        var_init = ir.Assign(iv, start, pos)
        initializers.append(var_init)
        induction_step = ir.BinOp(iv, step, "+")
        induction_assign = ir.Assign(iv, induction_step, pos)
        repl_body.append(induction_assign)
        cond_break = ir.break_if_matches(expr=stop, cond=True)
        repl_body.append(cond_break)

    # make simplified loop header
    (start, step), (stop, target) = loop_index_params

    iterable = ir.AffineSeq(star, stop, step)
    repl_header = ir.ForLoop(target, iterable, repl_body, pos)
    return repl_header


def make_single_index_loop(self, header: ir.ForLoop, symbols):
    """

        Make loop interval of the form (start, stop, step).

        This tries to find a safe method of calculation.

        This assumes (with runtime verification if necessary)
        that 'stop - start' will not overflow.

        References:
            LIVINSKII et. al, Random Testing for C and C++ Compilers with YARPGen
            Dietz et. al, Understanding Integer Overflow in C/C++
            Bachmann et. al, Chains of Recurrences - a method to expedite the evaluation of closed-form functions
            https://gcc.gnu.org/onlinedocs/gcc/Integer-Overflow-Builtins.html
            https://developercommunity.visualstudio.com/t/please-implement-integer-overflow-detection/409051
            https://numpy.org/doc/stable/user/building.html

    """
    # Assume each target appears exactly once, since we can always transform
    # to this case.

    # Find the (start, step) pair for each iterable.

    by_start_step = defaultdict(set)
    by_iterable = {}
    for _, iterable in unpack_iterated(header):
        start, stop, step = interval_from_iterable(iterable)
        key = (start, step)
        by_start_step[key].add(stop)
        by_iterable[iterable] = key

    by_start_step_opt = {}
    loop_params = {}

    # make index variables

    for key, stop in by_start_step.items():
        stop = ir.Min(stop)
        stop = simplify_commutative_min_max(stop)
        index_var = symbols.register_unique_name(prefix="i")
        loop_params[key] = stop, iv

    # Getting well defined explicit iteration counts has a lot of edge cases, including
    # division by zero and possible arithmetic overflow, where checks for arithmetic overflow
    # are compiler specific. In addition, there's the cost of potentially computing division for low trip counts
    # on multiple step sizes. It's easier just to take the naive approach and put all additional tests at the beginning
    # of the loop body.

    lp = iter(loop_params.items())
    loop_index_params = next(lp)

    initializers = []

    # anything other than the primary loop index is tested at the end of the loop body
    # This doesn't preserve initial testing order.

    # epilogue = []
    repl_body = []
    pos = header.pos
    # now set up the initial assignments

    for target, iterable in unpack_iterated(header):
        key = by_iterable[iterable]
        _, iv = loop_params[key]
        if isinstance(iterable, ir.AffineSeq):
            value = iv
        else:
            value = ir.Subscript(iterable, iv)
        assign = ir.Assign(target, value)
        repl_body.append(assign)

    repl_body.extend(header.body)

    for (start, step), (stop, iv) in lp:
        var_init = ir.Assign(iv, start, pos)
        initializers.append(var_init)
        induction_step = ir.BinOp(iv, step, "+")
        induction_assign = ir.Assign(iv, induction_step, pos)
        repl_body.append(induction_assign)
        cond_break = ir.break_if_matches(expr=stop, cond=True)
        repl_body.append(cond_break)

    # make simplified loop header
    (start, step), (stop, target) = loop_index_params

    iterable = ir.AffineSeq(star, stop, step)
    repl_header = ir.ForLoop(target, iterable, repl_body, pos)
    return repl_header
