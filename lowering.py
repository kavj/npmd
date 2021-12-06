import operator
import typing

import numpy as np
from collections import defaultdict
from functools import singledispatchmethod, singledispatch

import ir
import type_resolution as tr
from errors import CompilerError
from pretty_printing import pretty_formatter
from symbol_table import symbol_table
from utils import is_truth_test, wrap_constant, unpack_iterated, is_numeric_constant
from visitor import ExpressionVisitor, ExpressionTransformer, StmtTransformer, walk

unaryops = {ir.USUB: operator.neg,
            ir.UNOT: operator.inv,
            }

binops = {ir.ADD: operator.add,
          ir.SUB: operator.sub,
          ir.MULT: operator.mul,
          ir.TRUEDIV: operator.truediv,
          ir.FLOORDIV: operator.floordiv,
          ir.MOD: operator.mod,
          ir.POW: operator.pow,
          ir.MATMULT: operator.matmul,
          ir.EQ: operator.eq,
          ir.NE: operator.ne,
          ir.LT: operator.lt,
          ir.LE: operator.le,
          ir.GT: operator.gt,
          ir.GE: operator.ge,
          ir.LSHIFT: operator.lshift,
          ir.RSHIFT: operator.rshift,
          ir.BITAND: operator.and_,
          ir.BITOR: operator.or_,
          ir.BITXOR: operator.xor,
          }


const_folders = {
    ir.ADD: np.add,
    ir.SUB: np.subtract,
    ir.MULT: np.multiply,
    ir.TRUEDIV: np.true_divide,
    ir.FLOORDIV: np.floor_divide,
    ir.MOD: np.mod,
    ir.POW: np.power,
    ir.LSHIFT: np.left_shift,
    ir.RSHIFT: np.right_shift,
    ir.BITOR: np.bitwise_or,
    ir.BITAND: np.bitwise_and,
    ir.BITXOR: np.bitwise_xor,
    ir.UNOT: np.bitwise_not
}


def fold_constant_binop(node: ir.BinOp):
    if isinstance(node.left, ir.Constant) and isinstance(node.right, ir.Constant):
        assert not isinstance(node.left.value, np.bool_)
        assert not isinstance(node.left.value, np.bool_)
        if isinstance(node, (ir.TRUEDIV, ir.FLOORDIV)):
            if node.right == ir.Zero:
                msg = f"Divide by zero error {node}."
                raise CompilerError(msg)
        folder = const_folders[type(node)]
        return ir.Constant(folder(node.left.value, node.right.value))
    return node


@singledispatch
def is_unsafe_arithmetic(node):
    raise NotImplementedError


def is_constant_expr(expr):
    return all(isinstance(c, ir.Constant) for c in walk(expr))


def serialize_reduction(reduc: typing.Union[ir.MinReduction, ir.MaxReduction], syms: symbol_table, pos: ir.Position):
    """
    Turn a reduction into a series of single statement pairwise operations.

    :param reduc: initial reduction
    :syms: Symbol table for generating intermediates
    :return:  statements and final expression
    """

    op = ir.Min if isinstance(reduc, ir.MinReduction) else ir.Max
    values = set(reduc.subexprs)
    stmts = []

    if len(values) == 1:
        return values, stmts

    type_infer = tr.ExprTypeInfer(syms)
    tail = values.pop() if len(values) % 2 else None

    while len(values) > 2:
        value_iter = iter(values)
        repl_values = []
        for left, right in zip(value_iter, value_iter):
            expr = op(left, right)
            t = type_infer(expr)
            name = syms.make_unique_name_like(name="tmp", type_=t)
            assign = ir.Assign(target=name, value=expr, pos=pos)
            stmts.append(assign)
        values = repl_values

    out_expr = op(*values)

    if tail is not None:
        out_expr = op(out_expr, tail)

    return out_expr, stmts


class MinMaxSimplifier(ExpressionTransformer):

    def __init__(self, syms):
        self.syms = syms

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.MaxReduction):
        numeric = set()
        symbolic = set()
        for subexpr in node.subexprs:
            subexpr = self.visit(subexpr)
            if isinstance(subexpr, ir.Constant):
                numeric.add(subexpr.value)
            else:
                symbolic.add(subexpr)
        if numeric:
            numeric = wrap_constant(np.max(tuple(numeric)))
            if numeric == ir.NAN or not symbolic:
                return numeric
            symbolic.add(numeric)
        return ir.MaxReduction(symbolic)

    @visit.register
    def _(self, node: ir.MinReduction):
        numeric = set()
        symbolic = set()
        for subexpr in node.subexprs:
            subexpr = self.visit(subexpr)
            if isinstance(subexpr, ir.Constant):
                numeric.add(subexpr.value)
            else:
                symbolic.add(subexpr)
        if numeric:
            numeric = wrap_constant(np.min(tuple(numeric)))
            if numeric == ir.NAN or not symbolic:
                return numeric
            symbolic.add(numeric)
        return ir.MinReduction(*symbolic)

    @visit.register
    def _(self, node: ir.Min):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if left == right:
            return left
        elif ir.NAN in (left, right):
            # follow numpy rules
            return ir.NAN
        return ir.Min(left, right)

    @visit.register
    def _(self, node: ir.Max):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if left == right:
            return left
        elif ir.NAN in (left, right):
            return ir.NAN
        return ir.Max(left, right)


# Todo: A lot of this can't be done consistently without type information. Some folding of constants
#       is actually useful in determining the types that can hold a constant, but they shouldn't be
#       completely folded until we have type info. Otherwise we can end up using overflowing values
#       in a way that may differ from compiler sanitization flags.


def is_numeric_const_expr(expr):
    if isinstance(expr, ir.Expression):
        return all(isinstance(s, ir.Constant) and not s.is_predicate for s in expr.subexprs)
    elif is_numeric_constant(expr):
        return True
    return False


def unwrap_truth_tested(expr):
    """
    Extract truth tested operands. This helps limit isinstance checks for cases
    where an enclosing expression will refer to the truth test of expr, with or
    without an explicit TRUTH node wrapper.
    """
    if isinstance(expr, ir.TRUTH):
        expr = expr.operand
    return expr


class arithmetic_folding(ExpressionTransformer):
    """
    Non-recursive visitor for folding identity ops

    This does not respect unordered operands such as nans. It's primarily intended for integer
    arithmetic.

    """

    def __init__(self, syms):
        self.typing = tr.ExprTypeInfer(syms)

    @singledispatchmethod
    def visit(self, expr):
        return super().visit(expr)

    @visit.register
    def _(self, node: ir.POW):
        repl = ir.POW(self.visit(node.left), self.visit(node.right))
        repl = fold_constant_binop(repl)
        if isinstance(repl, ir.Constant):
            return repl
        t = self.typing.visit(repl)
        cls = ir.by_ir_type[t]
        if repl.right == ir.Zero:
            # forward type information through cast
            # wrap constant should handle this explicitly
            return cls(1)
        elif repl.left == ir.Zero:
            # only fold this case if it's provably safe
            if isinstance(repl.right, ir.Constant) and repl.right.value > 0:
                # get type from left
                return cls(0)
        elif repl.right == ir.One:
            # this needs to deal with casting
            return repl.left
        elif repl.right == ir.Constant(2):
            # again, have to deal with the implied cast
            return ir.MULT(repl.left, repl.right)
        elif repl.right == ir.Constant(0.5):
            # check for square root
            return ir.Sqrt(repl.left)
        return ir.POW(repl.left, repl.right)

    @visit.register
    def _(self, node: ir.ADD):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if right == ir.Zero:
            return left
        elif isinstance(right, ir.USUB):
            return ir.SUB(left, right.operand)
        elif isinstance(left, ir.USUB):
            return ir.SUB(right, left.operand)
        return ir.ADD(left, right)

    @visit.register
    def _(self, node: ir.SUB):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if left == ir.Zero:
            if isinstance(right, ir.USUB):
                return right.operand
            else:
                return ir.USUB(right)
        elif isinstance(right, ir.USUB):
            return ir.ADD(left, right.operand)
        elif right == ir.Zero:
            return left
        return ir.SUB(left, right)

    @visit.register
    def _(self, node: ir.MULT):
        left = self.visit(node.left)
        right = self.visit(node.right)
        negative_one = ir.Constant(-1)
        if left == ir.Zero or right == ir.Zero:
            return ir.Zero
        elif left == ir.One:
            return right
        elif right == ir.One:
            return left
        elif left == negative_one:
            if isinstance(right, ir.USUB):
                return right.operand
            else:
                return ir.USUB(right)
        elif right == negative_one:
            if isinstance(left, ir.USUB):
                return left.operand
            else:
                return ir.USUB(left)
        return ir.MULT(left, right)

    @visit.register
    def _(self, node: ir.FLOORDIV):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if right == ir.Zero:
            msg = f"Divide by zero error in expression {node}."
            raise CompilerError(msg)
        elif left == ir.Zero or right == ir.One:
            # cast appropriately
            return left
        return ir.FLOORDIV(left, right)

    @visit.register
    def _(self, node: ir.TRUEDIV):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if right == ir.Zero:
            msg = f"Divide by zero error in expression {node}."
            raise CompilerError(msg)
        elif right == ir.One:
            # need to ensure this is cast to float

            pass
        return ir.TRUEDIV(left, right)

    @visit.register
    def _(self, node: ir.UnaryOp):
        if isinstance(node.operand, ir.UnaryOp) and isinstance(node.operand, type(node)):
            return node.operand.operand
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
        operand, = node.subexprs
        operand = self.visit(operand)
        if isinstance(operand, ir.NOT):
            operand, = operand.subexprs
            return operand
        elif isinstance(operand, ir.EQ):
            return ir.NE(*operand.subexprs)
        elif isinstance(operand, ir.NE):
            return ir.EQ(*operand.subexprs)
        # other operators are only safe with integer type operands
        return ir.NOT(operand)

    @visit.register
    def _(self, node: ir.TRUTH):
        # This will leave truth casts on constant integers
        # and floats, since the only gain there is a loss
        # of clarity.
        if is_truth_test(node.operand):
            node = node.operand
        return node

    @visit.register
    def _(self, node: ir.Select):
        if node.on_true == node.on_false:
            return node.on_true
        elif node.predicate.constant:
            return node.on_true if operator.truth(node.predicate) else node.on_false
        predicate, on_true, on_false = node.subexprs
        if on_true == on_false:
            return on_true
        elif predicate.constant:
            return on_true if operator.truth(predicate) else on_false
        if isinstance(predicate, (ir.LT, ir.LE)):
            left, right = predicate.subexprs
            if on_true == left and on_false == right:
                return ir.Min(on_true, on_false)
            elif on_false == right and on_true == left:
                # This is almost negated. The issue is if in the destination assembly:
                #
                #     min(a,b) is implemented as a if a <= b else b
                #     max(a,b) is implemented as a if a >= b else b
                #
                #  which is common, we reverse operand order to properly catch unordered cases
                #  This does not follow Python's min/max conventions, which are too error prone.
                #  Those can arbitrarily propagate or suppress nans as a side effect of
                #  determining type from the leading operand.
                return ir.Max(on_false, on_true)
        elif isinstance(predicate, (ir.GT, ir.GE)):
            if on_true == predicate.left and on_false == predicate.right:
                return ir.Max(on_true, on_false)
            elif on_true == predicate.right and on_false == predicate.left:
                # right if left < right else left
                return ir.Min(on_false, on_true)
        return node


class IntervalBuilder(ExpressionVisitor):

    @singledispatchmethod
    def visit(self, iterable):
        msg = f"No method to generate interval for {iterable} of type {type(iterable)}."
        raise TypeError(msg)

    @visit.register
    def _(self, iterable: ir.AffineSeq):
        return iterable.start, iterable.stop, iterable.step

    @visit.register
    def _(self, iterable: ir.NameRef):
        return ir.Zero, ir.SingleDimRef(iterable, ir.Zero), ir.One

    @visit.register
    def _(self, iterable: ir.Subscript):
        if isinstance(iterable.index, ir.Slice):
            slice_ = iterable.index
            start = slice_.start
            if start is None:
                start = ir.Zero
            stop = ir.SingleDimRef(iterable.value, ir.Zero)
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


def _compute_iter_count(diff, step):
    # Todo: may insert an extra round of constant folding here..
    if step == ir.Zero:
        msg = "Zero step loop iterator encountered."
        raise CompilerError(msg)
    elif step == ir.One:
        count = diff
    else:
        on_false = ir.FLOORDIV(diff, step)
        modulo = ir.MOD(diff, step)
        on_true = ir.ADD(on_false, ir.One)
        count = ir.Select(predicate=modulo, on_true=on_true, on_false=on_false)
    return count


def _find_range_intersection(by_step):
    if len(by_step) == 0:
        return set()
    diff_sets = iter(by_step.value())
    initial = next(diff_sets)
    for d in diff_sets:
        initial.intersection_update(d)
    return initial


def _find_shared_interval(intervals, syms):
    starts = set()
    stops = set()
    steps = set()
    for start, stop, step in intervals:
        starts.add(start)
        stops.add(stop)
        steps.add(step)

    # enumerate doesn't declare a bound, so it shows up as None
    stops.discard(None)
    simplify_min_max = MinMaxSimplifier(syms)
    simplify_expr = arithmetic_folding(syms)

    if len(steps) == 1:
        # If there's only one step size, we can
        # avoid explicitly computing iteration count
        step = steps.pop()
        step = simplify_expr(step)
        if len(starts) == 1:
            start = starts.pop()
            if len(stops) == 1:
                stop = stops.pop()
                simplify_expr(stop)
            else:
                stop = ir.MinReduction(*(simplify_expr(s) for s in stops))
                stop = simplify_min_max(stop)
            return start, stop, step
        elif len(stops) == 1:
            stop = stops.pop()
            stop = simplify_expr(stop)
            start = {simplify_expr(s) for s in starts}
            start = ir.MaxReduction(start)
            start = simplify_min_max(start)
            diff = ir.SUB(stop, start)
            diff = simplify_expr(diff)
            return ir.Zero, diff, step

    # collect steps to minimize division and modulo ops required
    by_step = defaultdict(set)
    by_diff = defaultdict(set)
    for start, stop, step in intervals:
        diff = ir.SUB(stop, start)
        diff = simplify_expr(diff)
        by_step[step].add(diff)

    for step, diffs in by_step.items():
        by_step[frozenset(diffs)].add(step)

    # combine steps if possible
    for diff, steps in by_diff.items():
        if len(steps) != 1:
            # remove combinable entries
            for step in steps:
                by_step.pop(step)
            steps = ir.MaxReduction(steps)
            steps = simplify_min_max(steps)
            by_step[steps].update(diff)

    by_step_refined = {}

    for step, diffs in by_step.items():
        diffs = ir.MinReduction(diffs)
        diffs = simplify_min_max(diffs)
        by_step_refined[step] = diffs

    # Now compute explicit counts, since we don't use explicit dependency testing
    # this isn't a big deal

    counts = set()
    for step, diff in by_step_refined.items():
        count = _compute_iter_count(diff, step)
        count = simplify_expr(count)
        counts.add(count)
    counts = ir.MinReduction(counts)
    counts = simplify_min_max(counts)
    return ir.Zero, counts, ir.One


def make_single_index_loop(header: ir.ForLoop, symbols):
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

    by_iterable = {}
    intervals = set()
    interval_from_iterable = IntervalBuilder()
    for _, iterable in unpack_iterated(header.target, header.iterable):
        interval = interval_from_iterable(iterable)
        by_iterable[iterable] = interval
        intervals.add(interval)

    loop_start, loop_stop, loop_step = _find_shared_interval(intervals, symbols)
    loop_expr = ir.AffineSeq(loop_start, loop_stop, loop_step)
    # Todo: would be good to find a default index type
    loop_counter = symbols.make_unique_name_like("i", type_=ir.Int64)
    body = []
    pos = header.pos
    simplify_expr = arithmetic_folding(symbols)

    for target, iterable in unpack_iterated(header.target, header.iterable):
        (start, _, step) = by_iterable[iterable]
        assert step == loop_step
        assert (start == loop_start) or (loop_start == ir.Zero)
        if step == loop_step:
            if start == loop_start:
                index = loop_counter
            else:
                assert loop_start == ir.Zero
                index = ir.ADD(loop_counter, start)
        else:
            # loop counter must be normalized
            assert loop_start == ir.Zero
            assert loop_step == ir.One
            index = ir.MULT(step, loop_counter)
            if start != ir.Zero:
                index = ir.ADD(start, index)

        index = simplify_expr(index)
        value = index if isinstance(iterable, ir.AffineSeq) else ir.Subscript(iterable, index)
        assign = ir.Assign(target, value, pos)
        body.append(assign)

    # Todo: this doesn't hoist initial setup
    body.extend(header.body)
    repl = ir.ForLoop(loop_counter, loop_expr, body, pos)
    return repl


class loop_lowering(StmtTransformer):

    def __init__(self, symbols):
        self.symbols = symbols

    def __call__(self, node):
        return self.visit(node)

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.ForLoop):
        interm = make_single_index_loop(node, self.symbols)
        body = self.visit(interm.body)
        repl = ir.ForLoop(interm.target, interm.iterable, body, node.pos)
        return repl


class remove_subarray_refs(StmtTransformer):

    def __init__(self, syms):
        self.infer_type = tr.ExprTypeInfer(syms)

    def __call__(self, node):
        return self.visit(node)

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.Assign):
        target = node.target
        value = node.value
        target_type = self.infer_type(node.target)
        if isinstance(target_type, ir.ArrayType) and isinstance(value, (ir.NameRef, ir.Subscript)):
            # remove simple view creation
            return
        return node

    @visit.register
    def _(self, node: list):
        stmts = []
        for stmt in node:
            repl = self.visit(stmt)
            if repl is not None:
                if isinstance(repl, list):
                    pf = pretty_formatter()
                    print([pf(r) for r in repl])
                stmts.append(repl)
        return stmts


class array_offset_maker(StmtTransformer):
    """
    Transforms subviews of arrays into offsets with respect to base.
    This should be making an offset variable for anything that doesn't subscript
    down to a scalar for now.
    """

    def _find_view_hierarchy(self, target):
        if isinstance(target, ir.Subscript):
            if isinstance(target.value, ir.Subscript):
                pf = pretty_formatter()
                msg = f"Unsupported nested subscripting {pf(target)}."
                raise CompilerError(msg)
            if isinstance(target.index, ir.Tuple):
                indices = [offset for offset in target.index.subexprs]
                indices.reverse()
            else:
                indices = [target.index]
            target = target.value
        else:
            indices = []
        # Now, find the parent array that name that yielded this
        base = target
        parent = self.expr_to_parent.get(target)
        #
        while parent is not None:
            indices.append(self.offsets[base])
            base = parent
            parent = self.expr_to_parent.get(parent)
        indices.reverse()
        # make expanded subscript from base ref to corresponding scalar
        if len(indices) == 1:
            indices = indices.pop()
        else:
            indices = ir.Tuple(*indices)
        return ir.Subscript(base, indices)

    def __init__(self, syms, expr_to_parent, offsets):
        self.infer_type = tr.ExprTypeInfer(syms)
        self.expr_to_parent = expr_to_parent
        self.offsets = offsets
        self.pf = pretty_formatter()

    def __call__(self, node):
        node = self.visit(node)
        return node

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.Assign):
        if isinstance(node.value, ir.Subscript):
            # index = node.value.index
            lhs_type = self.infer_type(node.target)
            rhs_type = self.infer_type(node.value)
            # check basic compatibility
            if isinstance(lhs_type, ir.ArrayType) and isinstance(rhs_type, ir.ArrayType):
                if lhs_type != rhs_type:
                    msg = f"Mismatched array refs ({self.pf(node.target)}, {self.pf(lhs_type)}) and ({self.pf(node.value)}, {self.pf(rhs_type)}) line: {node.pos.line_begin}."
                    raise CompilerError(msg)
            elif isinstance(lhs_type, ir.ScalarType) and isinstance(rhs_type, ir.ScalarType):
                if isinstance(node.value, ir.Subscript):
                    # this is an array to scalar subscript
                    full_subscript = self._find_view_hierarchy(node.value)
                    repl = ir.Assign(node.target, full_subscript, node.pos)
                    return repl
            else:
                # mixed
                msg = f"Incompatible assignment '{self.pf(node.target)}' of " \
                      f"'{self.pf(lhs_type)}' from '{self.pf(node.value)}' of type '{self.pf(rhs_type)}'."
                raise CompilerError(msg)
        return node
