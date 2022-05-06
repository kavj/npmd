import itertools
import math
import numbers
import operator
import typing

import numpy as np
from collections import defaultdict
from functools import singledispatchmethod, singledispatch

import ir
from errors import CompilerError
from type_checks import TypeHelper
from pretty_printing import pretty_formatter
from reductions import ExpressionMapper
from symbol_table import symbol_table
from utils import unpack_iterated
from visitor import StmtVisitor, StmtTransformer, walk



@singledispatch
def is_unsafe_arithmetic(node):
    raise NotImplementedError


def is_constant_expr(expr):
    return all(isinstance(c, ir.Constant) for c in walk(expr))


def has_nan(node: ir.Expression):
    for subexpr in node.subexprs:
        if isinstance(subexpr, ir.Constant):
            if np.isnan(subexpr.value):
                return True


def is_constant(node: ir.Expression):
    return all(isinstance(subexpr, ir.Constant) for subexpr in node.subexprs)


def fold_arithmetic_binop(folder, node):
    if has_nan(node):
        return ir.NAN
    if is_constant(node):
        return folder(node.left.value, node.right.value)
    return node


def fold_compare_op(folder, node, on_nan):
    if has_nan(node):
        return on_nan
    if is_constant(node):
        return folder(node.left.value, node.right.value)


def fold_bit_op(folder, node):
    if is_constant(node):
        for subexpr in node.subexprs:
            if not isinstance(subexpr.value, numbers.Integral):
                msg = f"Unsupported operand '{subexpr.value}' for bit field operator {type(node)}."
                raise CompilerError(msg)
        return np.folder(node.left.value, node.right.value)


@singledispatch
def fold_op(node):
    raise NotImplementedError


@fold_op.register
def _(node: ir.ValueRef):
    return node


@fold_op.register
def _(node: ir.ADD):
    return fold_arithmetic_binop(np.add, node)


@fold_op.register
def _(node: ir.SUB):
    return fold_arithmetic_binop(np.sub, node)


@fold_op.register
def _(node: ir.MULT):
    return fold_arithmetic_binop(np.multiply, node)


@fold_op.register
def _(node: ir.TRUEDIV):
    return fold_arithmetic_binop(np.true_divide, node)


@fold_op.register
def _(node: ir.FLOORDIV):
    return fold_arithmetic_binop(np.floor_divide, node)


@fold_op.register
def _(node: ir.MOD):
    return fold_arithmetic_binop(np.mod, node)


@fold_op.register
def _(node: ir.POW):
    # Todo: This needs cast nodes, so that something like a**2 comes out as np.float64(a * a)
    if is_constant(node):
        return np.power(node.left, node.right)
    base, exponent = node.subexprs
    if exponent == ir.Zero:
        # numpy has this taking precedence
        if isinstance(exponent.value, numbers.Integral):
            # only fold if no unsafe stuff
            return ir.One
    elif has_nan(node):
        return ir.NAN
    elif base == 0:
        if isinstance(exponent.value, numbers.Integral):
            return ir.Zero
    elif exponent == ir.Half:
        return ir.Sqrt(base)
    elif exponent == ir.One:
        return base
    elif exponent == ir.Two:
        return ir.MULT(base, base)
    return node


@fold_op.register
def _(node: ir.MATMULT):
    return node


@fold_op.register
def _(node: ir.EQ):
    return fold_compare_op(np.equal, node, ir.FALSE)


@fold_op.register
def _(node: ir.NE):
    return fold_compare_op(np.not_equal, node, ir.TRUE)


@fold_op.register
def _(node: ir.LT):
    return fold_compare_op(np.less, node, ir.FALSE)


@fold_op.register
def _(node: ir.LE):
    return fold_compare_op(np.less_equal, node, ir.FALSE)


@fold_op.register
def _(node: ir.GT):
    return fold_compare_op(np.greater, node, ir.FALSE)


@fold_op.register
def _(node: ir.GE):
    return fold_compare_op(np.greater_equal, node, ir.FALSE)


@fold_op.register
def _(node: ir.AND):
    subexprs = []
    for subexpr in node.subexprs:
        if isinstance(node, ir.Constant):
            if node.value == 0:
                return ir.FALSE
        else:
            # only add non-constant exprs
            subexprs.append(subexpr)
    if len(subexprs) == 0:
        # all True constants
        return ir.TRUE
    return ir.AND(*subexprs)


@fold_op.register
def _(node: ir.OR):
    subexpr = []
    for subexpr in node.subexprs:
        if isinstance(node, ir.Constant):
            if node.value != 0:
                return ir.TRUE
        else:
            subexprs.append(subexpr)
    if len(subexprs) == 0:
        # all False constants
        return ir.FALSE


@fold_op.register
def _(node: ir.XOR):
    if is_constant(node):
        accum = node.operands[0]
        for term in itertools.islice(node.operands, 1):
            accum = operator.xor(accum, term)
        return accum
    return node


@fold_op.register
def _(node: ir.LSHIFT):
    return fold_bit_op(np.left_shift, node)


@fold_op.register
def _(node: ir.RSHIFT):
    return fold_bit_op(np.right_shift, node)


@fold_op.register
def _(node: ir.BITOR):
    return fold_bit_op(np.bitwise_or, node)


@fold_op.register
def _(node: ir.BITAND):
    return fold_bit_op(np.bitwise_and, node)


@fold_op.register
def _(node: ir.BITXOR):
    return fold_bit_op(np.bitwise_xor, node)


@fold_op.register
def _(node: ir.UNOT):
    if is_constant(node):
        return ir.TRUE if node.operand == ir.Zero else ir.FALSE
    return node


@fold_op.register
def _(node: ir.USUB):
    if is_constant(node):
        return np.negative(node.operand.value)
    return node


@fold_op.register
def _(self, node: ir.TRUTH):
    # This will leave truth casts on constant integers
    # and floats, since the only gain there is a loss
    # of clarity.
    if isinstance(node, ir.Constant):
        if node.value == 0:
            return ir.FALSE
        else:
            return ir.TRUE
    return node


@fold_op.register
def _(self, node: ir.Select):
    if isinstance(node.predicate, ir.Constant):
        # partial fold if constant predicate
        if node.predicate.value:
            return node.on_true
        else:
            return node.on_false
    return node
    if is_constant(node):
        if node.predicate.value:
            return node.on_true
        else:
            return node.on_false
    # predicate is non-const. Check if coercible to Min or max
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


def simplify_expr(node: ir.Expression):
    rewritten = {}
    for subexpr in walk(node):
        if isinstance(subexpr, ir.Expression):
            # find any previously simplified
            rewrite = subexpr.reconstruct(*(rewritten.get(s, s) for s in subexpr.subexprs))
            # Now try to simplify the resulting expression
            rewritten[subexpr] = fold_op(rewrite)
    return rewritten.get(node, node)


class TargetCollector(StmtVisitor):

    def __init__(self):
        self.targets = defaultdict(list)

    @classmethod
    def run(cls, node):
        obj = cls()
        obj.visit(node)
        return obj.targets

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Assign):
        self.targets[node.target].append(node.value)

    @visit.register
    def _(self, node: ir.ForLoop):
        for target, iterable in unpack_iterated(node):
            self.targets[target].append(iterable)


def collect_views(node, typer: TypeHelper):
    targets = TargetCollector.run(node)
    views = {}
    others = {}
    for target, values in targets.items():
        t = typer.check_type(target)
        if isinstance(t, ir.ArrayType):
            if len(values) != 1:
                pf = pretty_formatter()
                msg = f"Sub-array views must be uniquely defined, received: {pf(target)} {values}."
                raise CompilerError(msg)
            views[target] = vales.pop()
        else:
            others[target] = values
    return views, others


def find_subscripts(expr: ir.Expression):
    return {e for e in walk(expr) if isinstance(e, ir.Subscript)}


def name_indices(target: ir.ValueRef, iterable: ir.ValueRef, clobbers: typing.Dict[ir.ValueRef, ir.ValueRef]):
    # Todo: debating ditching sub-array views in this manner
    if not isinstance(target, ir.NameRef):
        pf = pretty_formatter()
        ft = pf(target)
        fi = pf(iterable)
        msg = f"Binding subscripts in loop headers is unsupported: {ft} from {fi}."
        raise CompilerError(msg)
    if iterable in clobbers:
        pass


class IndexFreezer(StmtTransformer):
    def __init__(self, views: typing.Dict[ir.ValueRef, ir.ValueRef],
                 clobbers: typing.Dict[ir.ValueRef, ir.ValueRef],
                 typer: TypeHelper):
        self.views = views
        self.view_to_index_name = {}
        self.clobbers = clobbers
        self.typer = typer

    @classmethod
    def freeze(cls, node, views, clobbers, typer):
        visitor = cls()

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: list):
        stmts = []
        for stmt in node:
            repl = self.visit(stmt)
            if isinstance(repl, list):
                stmts.extend(repl)
            else:
                stmts.append(repl)

    @visit.register
    def _(self, node: ir.ForLoop):
        for target, iterable in unpack_iterated(node.target, node.iterable):
            t = self.typer.check_type(target)
            if isinstance(t, ir.ArrayType):
                assert isinstance(target, ir.NameRef)
                index_name = self.typer.declare_like(target, prefix='i')
                if target in self.view_to_index_name:
                    msg = f"Target array view {target} is not uniquely bound."
                    raise CompilerError(msg)
                self.view_to_index_name[target] = index_name

class LoopLower(StmtTransformer):

    def __init__(self, views: typing.Dict[ir.ValueRef, ir.ValueRef],
                 clobbers: typing.Dict[ir.ValueRef, ir.ValueRef],
                 typer: TypeHelper):
        self.views = views
        self.view_to_index = {}
        self.clobbers = clobbers
        self.typer = typer

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Assign):
        if node.target in self.views:
            if isinstance(node.value, ir.Subscript):
                # Ensure we can propagate
                index = node.value.index
                if any(subexpr in self.clobbers for subexpr in walk(index)):
                    repl_index = self.typer.declare_like(index)
                    index_assign = ir.Assign(repl_index, node.vale.value, node.pos)
                    repl_subscript = ir.Subscript(node.value.value, repl_index)
                    index_assign = ir.Assign(s, node.target, node.value)


def lower_loops(node, symbols: symbol_table):
    typer = TypeHelper(symbols)
    views = collect_views(node, typer)


class IntervalBuilder:

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

    type_infer = TypeHelper(syms)
    tail = values.pop() if len(values) % 2 else None

    while len(values) > 2:
        value_iter = iter(values)
        repl_values = []
        for left, right in zip(value_iter, value_iter):
            expr = op(left, right)
            t = type_infer.check_type(expr)
            name = syms.make_unique_name_like(name="tmp", type_=t)
            assign = ir.Assign(target=name, value=expr, pos=pos)
            stmts.append(assign)
        values = repl_values

    out_expr = op(*values)

    if tail is not None:
        out_expr = op(out_expr, tail)

    return out_expr, stmts


def unwrap_truth_tested(expr):
    """
    Extract truth tested operands. This helps limit isinstance checks for cases
    where an enclosing expression will refer to the truth test of expr, with or
    without an explicit TRUTH node wrapper.
    """
    if isinstance(expr, ir.TRUTH):
        expr = expr.operand
    return expr


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
    # simplify_min_max = MinMaxSimplifier(syms)

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
                # stop = simplify_min_max.visit(stop)
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
        interval = interval_from_iterable.visit(iterable)
        by_iterable[iterable] = interval
        intervals.add(interval)

    interval = _find_shared_interval(intervals, symbols)
    loop_start, loop_stop, loop_step = interval
    # find reductions
    seen = {}
    loop_expr = ir.AffineSeq(loop_start, loop_stop, loop_step)
    # Todo: would be good to find a default index type
    loop_counter = symbols.make_unique_name_like("i", type_=ir.int64)
    body = []
    pos = header.pos

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


class LoopLowering(StmtTransformer):

    def __init__(self, symbols: symbol_table):
        self.symbols = symbols

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.ForLoop):
        # make this loop into a single index
        rewrite = make_single_index_loop(node, self.symbols)
        if not isinstance(rewrite.iterable, ir.AffineSeq):
            raise ValueError(msg)
        body = self.visit(rewrite.body)
        repl = ir.ForLoop(rewrite.target, rewrite.iterable, body, rewrite.pos)
        return repl
