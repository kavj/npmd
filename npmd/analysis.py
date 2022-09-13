from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import fields
from functools import singledispatch
from typing import Iterable, List, Optional, Union

import npmd.ir as ir

from npmd.errors import CompilerError
from npmd.utils import unpack_iterated
from npmd.traversal import get_statement_lists, walk, walk_parameters

# Todo: stub
specialized = {ir.NameRef("print")}


def find_assigned_or_augmented(entry: Union[ir.Function, ir.ForLoop, ir.WhileLoop]):
    for stmt in get_statement_lists(entry):
        if isinstance(stmt, ir.ForLoop):
            for target, _ in unpack_iterated(stmt.target, stmt.iterable):
                yield target
        elif isinstance(stmt, (ir.Assign, ir.InPlaceOp)):
            if isinstance(stmt.target, ir.NameRef):
                yield stmt.target
            else:
                assert isinstance(stmt.target, ir.Subscript)
                yield stmt.target.value


def expression_strictly_contains(a: ir.ValueRef, b: ir.ValueRef):
    """
    returns True if evaluation of 'b' requires evaluation of 'a'.

    :param a:
    :param b:
    :return:
    """

    if not isinstance(b, ir.Expression) or a == b:
        return False
    for subexpr in walk(b):
        if subexpr == a:
            return True
    return False


def greatest_common_subexprs(a: ir.ValueRef, b: ir.ValueRef):
    """
    returns the broadest sub-expressions that are common to 'a' and 'b'.
    :param a:
    :param b:
    :return:
    """

    if a == b:
        return a

    subexprs_a = set(walk(a))
    subexprs_b = set(walk(b))

    # get subexpression overlap if any
    common = subexprs_a.intersection(subexprs_b)
    if not common:
        return common

    sub = set()
    for u in common:
        for v in common:
            if expression_strictly_contains(v, u):
                # remove all expressions which are sub-expressions of any common sub-expression
                sub.add(u)
    common.difference_update(sub)
    return common


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


def get_assign_counts(node: Union[ir.Function, ir.ForLoop, ir.WhileLoop]):
    """
    Count the number of times each variable is assigned within the function body
    :param node:
    :return:
    """
    assign_counts = Counter()
    for stmts in get_statement_lists(node):
        for stmt in stmts:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.target, ir.NameRef):
                assign_counts[stmt.target] += 1
            elif isinstance(stmt, ir.ForLoop):
                for target, _ in unpack_iterated(stmt.target, stmt.iterable):
                    # it's okay to count duplicates in unpacking
                    assign_counts[target] += 1
    return assign_counts


def find_index_parameters(node: ir.ForLoop):
    """

    :param node:
    :return:
    """
    # Todo: This should be able to handle more types
    index_params = set()
    for stmts in get_statement_lists(node):
        for stmt in stmts:
            for expr in extract_expressions(stmt):
                for subexpr in walk(expr):
                    if isinstance(subexpr, ir.Subscript):
                        index_params.update(walk_parameters(subexpr))
    return index_params


@singledispatch
def extract_expressions(node):
    msg = f'No method to extract expressions from {node}'
    raise TypeError(msg)


@extract_expressions.register
def _(node: ir.StmtBase):
    return
    yield


@extract_expressions.register
def _(node: ir.Assign):
    yield node.value
    if not isinstance(node.target, ir.NameRef):
        yield node.target


@extract_expressions.register
def _(node: ir.ForLoop):
    for target, iterable in unpack_iterated(node.target, node.iterable):
        yield iterable
        yield target


@extract_expressions.register
def _(node: ir.IfElse):
    yield node.test


@extract_expressions.register
def _(node: ir.InPlaceOp):
    yield node.value


@extract_expressions.register
def _(node: ir.SingleExpr):
    yield node.value


@extract_expressions.register
def _(node: ir.Return):
    yield node.value


@extract_expressions.register
def _(node: ir.WhileLoop):
    yield node.test


def extract_parameters(stmt: ir.StmtBase):
    for expr in extract_expressions(stmt):
        yield from walk_parameters(expr)


def statements_match(*stmts: Iterable[ir.StmtBase]):
    first = stmts[0]
    first_type = type(first)
    fields_first = fields(first)
    assert isinstance(first, ir.StmtBase)
    for stmt in stmts:
        if type(stmt) != first_type:
            return False
        if fields_first != fields(stmt):
            raise ValueError(f'field names do not match between "{first}" and "{stmt}"')
        # We have matching type and fields (second should always be true unless live objects updated).
        # Now we need to check if values other than positional information all match.
        # We're avoiding asdict here to avoid the use of deepcopy
        for f in fields_first:
            name = f.name
            if f.name == 'pos':
                continue
            if getattr(first, name) != getattr(stmt, name):
                return False


def get_possible_call_clobbers(stmt: ir.StmtBase):
    for expr in extract_expressions(stmt):
        for subexpr in walk(expr):
            if isinstance(subexpr, ir.Call):
                # anything that is passed as an argument could be augmented
                # This is really only a risk for array arguments, but we don't want
                # to force this to depend on argument type information
                yield from subexpr.args
