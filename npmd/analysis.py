from __future__ import annotations

from collections import Counter, deque
from contextlib import contextmanager
from typing import Iterable, List, Optional, Set, Tuple

import npmd.ir as ir

from npmd.errors import CompilerError
from npmd.utils import is_entry_point, unpack_iterated
from npmd.traversal import walk_nodes, walk_parameters

# Todo: stub
specialized = {ir.NameRef("print")}


class DeclTracker:
    """
    A class to track what should be declared at this point.
    """

    def __init__(self):
        self.declared = []

    @contextmanager
    def scope(self):
        s = set()
        self.declared.append(s)
        yield
        self.declared.pop()

    def innermost(self):
        return self.declared[-1]

    def declare(self, node: ir.NameRef):
        if self.is_undeclared(node):
            self.declared[-1].add(node)

    def is_declared(self, node: ir.NameRef):
        return any(node in d for d in reversed(self.declared))

    def is_undeclared(self, node: ir.NameRef):
        return not any(node in d for d in reversed(self.declared))

    def find_undeclared(self, expr: ir.ValueRef):
        # collects rather than raising
        return {p for p in walk_parameters(expr) if isinstance(p, ir.NameRef) and self.is_undeclared(p)}

    def check_undeclared(self, expr: ir.ValueRef, current_line: ir.Position):
        for param in walk_parameters(expr):
            if self.is_undeclared(param):
                msg = f'Possible unbound local "{param.name}" at line {current_line}.'
                raise CompilerError(msg)


def check_all_declared(block: List[ir.StmtBase], tracker: Optional[DeclTracker] = None):
    if tracker is None:
        tracker = DeclTracker()
    for stmt in block:
        current_line = stmt.pos.line_begin
        if isinstance(stmt, ir.IfElse):
            tracker.check_undeclared(stmt.test, current_line)
            with tracker.scope():
                check_all_declared(stmt.if_branch, tracker)
                bound_if = tracker.innermost()
            with tracker.scope():
                check_all_declared(stmt.else_branch, tracker)
                bound_else = tracker.innermost()
            tracker.innermost().update(bound_if.intersection(bound_else))
        elif isinstance(stmt, ir.ForLoop):
            with tracker.scope():
                for target, iterable in unpack_iterated(stmt.target, stmt.iterable):
                    tracker.check_undeclared(iterable, current_line)
                    tracker.declare(target)
                with tracker.scope():
                    check_all_declared(stmt.body, tracker)
        elif isinstance(stmt, ir.WhileLoop):
            tracker.check_undeclared(stmt.test, current_line)
            with tracker.scope():
                check_all_declared(stmt.body, tracker)
        elif isinstance(stmt, ir.Assign):
            tracker.check_undeclared(stmt.value, current_line)
            if isinstance(stmt.target, ir.NameRef):
                tracker.declare(stmt.target)
            else:
                tracker.check_undeclared(stmt.target, current_line)
        elif isinstance(stmt, ir.InPlaceOp):
            tracker.check_undeclared(stmt.value, current_line)
        elif isinstance(stmt, (ir.SingleExpr, ir.Return)):
            tracker.check_undeclared(stmt.value, current_line)


def compute_element_count(start: ir.ValueRef, stop: ir.ValueRef, step: ir.ValueRef):
    """
    Single method to compute the number of elements for an iterable expression.

    :param start:
    :param stop:
    :param step:
    :return:
    """

    if start == ir.Zero:
        diff = stop
    else:
        # It's safer to put the max around start, since this doesn't risk overflow
        diff = ir.SUB(stop, ir.MAX(ir.Zero, start))
    if step == ir.One:
        return diff
    base_count = ir.FLOORDIV(diff, step)
    test = ir.MOD(diff, step)
    fringe = ir.SELECT(predicate=test, on_true=ir.One, on_false=ir.Zero)
    count = ir.ADD(base_count, fringe)
    return count


def find_array_length_expression(node: ir.ValueRef) -> Optional[ir.ValueRef]:
    """
      Find an expression for array length if countable. Otherwise returns None.
      For example:
          a[a > 0] is not trivially countable.
          The predicate must be evaluated against the array
          to determine the number of elements.

          a[i::] is trivially countable. For i >= 0, len(a[i::]) == len(a) - i

      :return:
    """
    if isinstance(node, ir.Subscript):
        if isinstance(node.index, ir.Slice):
            index = node.index
            start = index.start
            stop = index.stop
            base_len = ir.SingleDimRef(node.value, dim=ir.Zero)
            if stop is not None and stop != base_len:
                stop = ir.MIN(stop, base_len)
            return compute_element_count(start, stop, index.step)
        else:
            # first dim removed
            return ir.SingleDimRef(node.value, dim=ir.One)
    elif isinstance(node, ir.NameRef):
        return ir.SingleDimRef(node, dim=ir.Zero)


def get_branch_predicate_pairs(node: ir.IfElse):
    """
    If this branch is perfectly nested, convert it to a case statement

    Note: this makes further if conversion much simpler
    Note: we may want to break along loop uniform conditions

    :param node:
    :return:
    """

    predicates = deque()

    # Note, we only care about the first true instance of any unique predicate
    # Any duplicates become dead branches, and we need to avoid overwriting their corresponding entries

    queued = [node]

    while queued:
        stmt = queued.pop()
        predicate = stmt.test
        predicates.append((predicate, stmt.if_branch, stmt.pos))
        if len(stmt.else_branch) == 1 and isinstance(stmt.else_branch[0], ir.IfElse):
            queued.append(stmt.else_branch[0])
        else:
            assert not queued
            predicates.append((None, stmt.else_branch, None))
    return predicates


def get_read_and_assigned(stmts: Iterable[ir.StmtBase]) -> Tuple[Set[ir.NameRef], Set[ir.NameRef], Set[ir.NameRef]]:
    uevs = set()
    assigned = set()
    referenced = set()
    for stmt in stmts:
        if isinstance(stmt, (ir.Assign, ir.InPlaceOp)):
            for name in walk_parameters(stmt.value):
                if name not in assigned:
                    uevs.add(name)
                referenced.add(name)
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.target, ir.NameRef):
                    assigned.add(stmt.target)
                else:
                    # ensure supported target
                    assert isinstance(stmt.target, ir.Subscript)
                    for name in walk_parameters(stmt.target):
                        if name not in assigned:
                            uevs.add(name)
                        referenced.add(name)
        elif isinstance(stmt, (ir.SingleExpr, ir.Return)):
            for name in walk_parameters(stmt.value):
                if name not in assigned:
                    uevs.add(name)
                referenced.add(name)
        elif is_entry_point(stmt):
            # single statement blocks
            if isinstance(stmt, (ir.IfElse, ir.WhileLoop)):
                for name in walk_parameters(stmt.test):
                    if name not in assigned:
                        uevs.add(name)
                    referenced.add(name)
            else:
                assert isinstance(stmt, ir.ForLoop)
                for target, value in unpack_iterated(stmt.target, stmt.iterable):
                    for p in walk_parameters(value):
                        if p not in assigned:
                            uevs.add(p)
                        referenced.add(p)
                    if isinstance(target, ir.NameRef):
                        assigned.add(target)
                    else:
                        for p in walk_parameters(target):
                            if p not in assigned:
                                uevs.add(p)
                            referenced.add(p)
    return assigned, referenced, uevs


def get_assign_counts(node: ir.Function):
    """
    Count the number of times each variable is assigned within the function body
    :param node:
    :return:
    """
    assign_counts = Counter()
    for stmt in walk_nodes(node.body):
        if isinstance(stmt, ir.Assign) and isinstance(stmt.target, ir.NameRef):
            assign_counts[stmt.target] += 1
        elif isinstance(stmt, ir.ForLoop):
            for target, _ in unpack_iterated(stmt.target, stmt.iterable):
                # it's okay to count duplicates in unpacking
                assign_counts[target] += 1
    return assign_counts
