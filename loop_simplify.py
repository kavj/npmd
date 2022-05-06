import operator

from analysis import find_unterminated_path
from collections import defaultdict
from functools import singledispatchmethod

import ir

from errors import CompilerError
from symbol_table import SymbolTable
from utils import unpack_iterated
from visitor import StmtTransformer


def remove_trailing_continues(node: list) -> list:
    """
    Remove continues that are the last statement along some execution path within the current
    enclosing loop.

    """

    if len(node) > 0:
        last = node[-1]
        if isinstance(last, ir.IfElse):
            if_branch = remove_trailing_continues(last.if_branch)
            else_branch = remove_trailing_continues(last.else_branch)
            if if_branch != last.if_branch or else_branch != last.else_branch:
                last = ir.IfElse(last.test, if_branch, else_branch, last.pos)
                # copy original
                node = node[:-1]
                node.append(last)
        elif isinstance(last, ir.Continue):
            node = node[:-1]
    return node


class NormalizePaths(StmtTransformer):
    """
    This is the tree version of control flow optimization.
    It removes any statements blocked by break, return, or continue
    and inlines paths as an alternative to explicit continue statements.

    """

    def __init__(self):
        self.innermost_loop = None
        self.body = None

    def __call__(self, node):
        repl = self.visit(node)
        return repl

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.ForLoop):
        body = self.visit(node.body)
        body = remove_trailing_continues(body)
        if body != node.body:
            node = ir.ForLoop(node.target, node.iterable, body, node.pos)
        return node

    @visit.register
    def _(self, node: ir.WhileLoop):
        test = node.test
        if test.constant:
            if not operator.truth(test):
                # return None if the loop body is unreachable.
                return
            body = self.visit(node.body)
            body = remove_trailing_continues(body)
            if body != node.body:
                node = ir.WhileLoop(node.test, body, node.pos)
        return node

    @visit.register
    def _(self, node: ir.IfElse):
        if_branch = self.visit(node.if_branch)
        else_branch = self.visit(node.else_branch)
        if not (if_branch or else_branch):
            return
        node = ir.IfElse(node.test, if_branch, else_branch, node.pos)
        return node

    @visit.register
    def _(self, node: list):
        repl = []
        append_to = repl
        for stmt in node:
            if isinstance(stmt, ir.IfElse):
                if stmt.test.constant:
                    live_branch = stmt.if_branch if operator.truth(stmt.test) else stmt.else_branch
                    live_branch = self.visit(live_branch)
                    extendable_path = find_unterminated_path(live_branch)
                    append_to.extend(live_branch)
                    if extendable_path is not live_branch:
                        if extendable_path is None:
                            break
                        else:  # extendable path exists and is distinct from the inlined branch
                            append_to = extendable_path
                else:
                    stmt = self.visit(stmt)
                    if stmt is None:
                        continue  # doesn't execute anything
                    append_to.append(stmt)
                    if_path = find_unterminated_path(stmt.if_branch)
                    else_path = find_unterminated_path(stmt.else_branch)
                    if if_path is None and else_path is None:
                        break  # remaining statements are unreachable
                    elif if_path is None:
                        append_to = else_path
                    elif else_path is None:
                        append_to = if_path
            else:
                stmt = self.visit(stmt)
                if stmt is not None:
                    append_to.append(stmt)
                    if isinstance(stmt, (ir.Break, ir.Continue, ir.Return)):
                        break  # remaining statements are unreachable
        return repl


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
                stop = ir.MIN(stop, slice_.stop)
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
        count = ir.SELECT(predicate=modulo, on_true=on_true, on_false=on_false)
    return count


def _find_range_intersection(by_step):
    if len(by_step) == 0:
        return set()
    diff_sets = iter(by_step.value())
    initial = next(diff_sets)
    for d in diff_sets:
        initial.intersection_update(d)
    return initial


def _find_shared_interval(intervals):
    starts = set()
    stops = set()
    steps = set()
    for start, stop, step in intervals:
        starts.add(start)
        stops.add(stop)
        steps.add(step)

    # enumerate doesn't declare a bound, so it shows up as None
    stops.discard(None)

    if len(steps) == 1:
        # If there's only one step size, we can
        # avoid explicitly computing iteration count
        step = steps.pop()
        if len(starts) == 1:
            start = starts.pop()
            if len(stops) == 1:
                stop = stops.pop()
            else:
                stop = ir.MinReduction(*stops)
            return start, stop, step
        elif len(stops) == 1:
            stop = stops.pop()
            start = ir.MaxReduction(starts)
            diff = ir.SUB(stop, start)
            return ir.Zero, diff, step

    # collect steps to minimize division and modulo ops required
    by_step = defaultdict(set)
    by_diff = defaultdict(set)
    for start, stop, step in intervals:
        diff = ir.SUB(stop, start)
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
            by_step[steps].update(diff)

    by_step_refined = {}

    for step, diffs in by_step.items():
        diffs = ir.MinReduction(diffs)
        by_step_refined[step] = diffs

    # Now compute explicit counts, since we don't use explicit dependency testing
    # this isn't a big deal

    counts = set()
    for step, diff in by_step_refined.items():
        count = _compute_iter_count(diff, step)
        counts.add(count)
    counts = ir.MinReduction(counts)
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

    interval = _find_shared_interval(intervals)
    loop_start, loop_stop, loop_step = interval
    # find reductions
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

        value = index if isinstance(iterable, ir.AffineSeq) else ir.Subscript(iterable, index)
        assign = ir.Assign(target, value, pos)
        body.append(assign)

    # Todo: this doesn't hoist initial setup
    body.extend(header.body)
    repl = ir.ForLoop(loop_counter, loop_expr, body, pos)
    return repl


class LoopLowering(StmtTransformer):

    def __init__(self, symbols: SymbolTable):
        self.symbols = symbols

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.ForLoop):
        # make this loop into a single index
        rewrite = make_single_index_loop(node, self.symbols)
        body = self.visit(rewrite.body)
        repl = ir.ForLoop(rewrite.target, rewrite.iterable, body, rewrite.pos)
        return repl
