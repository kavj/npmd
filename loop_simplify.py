from collections import defaultdict
from functools import singledispatchmethod
from typing import List, Union

import ir

from analysis import is_terminated
from errors import CompilerError
from symbol_table import SymbolTable
from type_checks import contains_stmt_types
from utils import unpack_iterated


def remove_unreachable(body: List[ir.StmtBase]):
    # first find trivial cases
    repl = []

    for stmt in body:
        if isinstance(stmt, ir.ForLoop):
            loop_body = remove_unreachable(stmt.body)
            # old code removed more aggressively, but this case is trivial
            if loop_body and isinstance(loop_body[-1], ir.Continue):
                loop_body.pop()
            stmt = ir.ForLoop(stmt.target, stmt.iterable, loop_body, stmt.pos)
        elif isinstance(stmt, ir.WhileLoop):
            loop_body = remove_unreachable(stmt.body)
            if loop_body and isinstance(loop_body[-1], ir.Continue):
                loop_body.pop()
            stmt = ir.WhileLoop(stmt.test, loop_body, stmt.pos)
        elif isinstance(stmt, ir.IfElse):
            if_branch = remove_unreachable(stmt.if_branch)
            else_branch = remove_unreachable(stmt.else_branch)
            stmt = ir.IfElse(stmt.test, if_branch, else_branch, stmt.pos)
        repl.append(stmt)
        if is_terminated(stmt):
            break
    return repl


def get_iterator_access_pattern(node: Union[ir.AffineSeq, ir.NameRef, ir.Subscript]):
    if isinstance(node, ir.AffineSeq):
        return node.start, node.stop, node.step
    elif isinstance(node, ir.NameRef):
        return ir.Zero, ir.SingleDimRef(node, ir.Zero), ir.One
    elif isinstance(node, ir.Subscript):
        if isinstance(node.index, ir.Slice):
            slice_ = node.index
            start = slice_.start
            if start is None:
                start = ir.Zero
            stop = ir.SingleDimRef(node.value, ir.Zero)
            if slice_.stop is not None:
                stop = ir.MIN(stop, slice_.stop)
            step = slice_.step
            if step is None:
                step = ir.One
        else:
            start = ir.Zero
            stop = ir.SingleDimRef(node, ir.One)
            step = ir.One
        return start, stop, step
    else:
        msg = f"No method to generate interval for {node} of type {type(node)}."
        raise TypeError(msg)


def _compute_iter_count(diff: ir.ValueRef, step: ir.ValueRef):
    # Todo: replace with compute_element_count from analysis
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
    for _, iterable in unpack_iterated(header.target, header.iterable):
        interval = get_iterator_access_pattern(iterable)
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


class LoopLowering:

    def __init__(self, symbols: SymbolTable):
        self.symbols = symbols

    @singledispatchmethod
    def visit(self, node):
        msg = f'No method to lower node "{node}".'
        raise TypeError(msg)

    @visit.register
    def _(self, node: ir.Function):
        if not contains_stmt_types(node.body, (ir.ForLoop,)):
            return node
        body = self.visit(node.body)
        return ir.Function(node.name, node.args, body)

    @visit.register
    def _(self, node: list):
        return [self.visit(stmt) for stmt in node]

    @visit.register
    def _(self, node: ir.StmtBase):
        return node

    @visit.register
    def _(self, node: ir.IfElse):
        if_branch = self.visit(node.if_branch)
        else_branch = self.visit(node.else_branch)
        return ir.IfElse(node.test, if_branch, else_branch, node.pos)

    @visit.register
    def _(self, node: ir.WhileLoop):
        body = self.visit(node.body)
        return ir.WhileLoop(node.test, body, node.pos)

    @visit.register
    def _(self, node: ir.ForLoop):
        # make this loop into a single index
        rewrite = make_single_index_loop(node, self.symbols)
        body = self.visit(rewrite.body)
        repl = ir.ForLoop(rewrite.target, rewrite.iterable, body, rewrite.pos)
        return repl
