from __future__ import annotations
import numpy as np

from collections import defaultdict, Counter
from contextlib import contextmanager
from functools import singledispatch, singledispatchmethod
from typing import List, Optional, Set

import ir

from errors import CompilerError
from symbol_table import SymbolTable
from type_checks import contains_stmt_types, TypeHelper
from utils import is_entry_point, unpack_iterated
from traversal import depth_first_sequence_blocks, walk_parameters

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

    def check_undeclared(self, expr: ir.ValueRef, current_line):
        for param in walk_parameters(expr):
            if self.is_undeclared(param):
                msg = f'Possible unbound local "{param.name}" at line {current_line}.'
                raise CompilerError(msg)


def count_clobbers(block: List[ir.StmtBase]):
    clobbers = Counter()
    for index, stmt in enumerate(block):
        if isinstance(stmt, ir.Assign) and isinstance(stmt.target, ir.NameRef):
            clobbers[stmt.target] += 1
        elif isinstance(stmt, ir.ForLoop):
            assert index == 0
            for target, _ in unpack_iterated(stmt.target, stmt.iterable):
                clobbers[target] += 1
                break
    return clobbers


def count_assigning_blocks(node: List[ir.StmtBase]):
    """
    This counts the number of blocks that bind a particular name.
    Anything with exactly one assigning block that does not appear in arguments can be declared
    wherever it's first assigned.
    :param node:
    :return:
    """

    seen = set()
    counts = Counter()
    for block in depth_first_sequence_blocks(node):
        for stmt in block:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.target, ir.NameRef):
                seen.add(stmt.target)
        for name in seen:
            counts[name] += 1
        seen.clear()
    return counts


def has_nan(node: ir.Expression):
    for subexpr in node.subexprs:
        if isinstance(subexpr, ir.CONSTANT):
            if np.isnan(subexpr.value):
                return True


def get_possible_aliasing(stmts: List[ir.StmtBase], symbols: SymbolTable):
    """
    Get sub-array references that may be linked to an array name
    :param stmts:
    :param symbols:
    :return:
    """
    aliasing = defaultdict(set)
    for stmt in stmts:
        if isinstance(stmt, ir.Assign):
            if isinstance(stmt.target, ir.NameRef) and isinstance(stmt.value, ir.Subscript):
                if isinstance(symbols.check_type(stmt.target), ir.ArrayType):
                    aliasing[stmt.target].add(stmt.value)
                    aliasing[stmt.value].add(stmt.target)
    return aliasing


@singledispatch
def is_terminated(node):
    raise TypeError


@is_terminated.register
def _(node: ir.StmtBase):
    return isinstance(node, (ir.Break, ir.Continue, ir.Return))


@is_terminated.register
def _(node: ir.IfElse):
    return is_terminated(node.if_branch) and is_terminated(node.else_branch)


@is_terminated.register
def _(node: ir.WhileLoop):
    if not isinstance(node.test, ir.CONSTANT):
        return False
    if node.test.value:
        return contains_stmt_types(node.body, (ir.Break,))


@is_terminated.register
def _(body: list):
    for stmt in reversed(body):
        if isinstance(stmt, (ir.Break, ir.Continue, ir.Return)):
            return True
        elif isinstance(stmt, (ir.IfElse, ir.WhileLoop)):
            if is_terminated(stmt):
                return True
    return False


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


def check_all_paths_return(stmts: List[ir.StmtBase]):
    for stmt in stmts:
        if isinstance(stmt, ir.Return):
            return True
        elif isinstance(stmt, ir.IfElse):
            if check_all_paths_return(stmt.if_branch) and check_all_paths_return(stmt.else_branch):
                return True
        if isinstance(stmt, ir.WhileLoop) and is_terminated(stmt):
            return True
    return False


def compute_element_count(start: ir.ValueRef, stop: ir.ValueRef, step: ir.ValueRef, typer: TypeHelper):
    """
    Single method to compute the number of elements for an iterable expression.

    :param start:
    :param stop:
    :param step:
    :return:
    """

    diff = ir.SUB(stop, start)
    base_count = (ir.FLOORDIV(diff, step))
    test = ir.MOD(diff, step)
    fringe = ir.SELECT(predicate=test, on_true=ir.One, on_false=ir.Zero)
    count = ir.ADD(base_count, fringe)
    return count


def find_array_length_expression(node: ir.ValueRef, typer: TypeHelper) -> Optional[ir.ValueRef]:
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
            if index.stop is None:
                stop = ir.SingleDimRef(node.value, dim=ir.Zero)
            else:
                stop = ir.MIN(index.stop, ir.SingleDimRef(node.value, dim=ir.Zero))
            step = index.step
            return compute_element_count(start, stop, step, typer)
        else:
            # first dim removed
            return ir.SingleDimRef(node.value, dim=ir.One)
    elif isinstance(node, ir.NameRef):
        return ir.SingleDimRef(node, dim=ir.Zero)
    # other cases handled by analyzing sub-expressions and broadcasting constraints


def get_read_first(stmt: ir.StmtBase, clobbers: Set[ir.NameRef]):
    if isinstance(stmt, (ir.Assign, ir.InPlaceOp)):
        # Todo: need some other checks for unsupported tuple patterns
        for name in walk_parameters(stmt.value):
            if name not in clobbers:
                yield name
        if isinstance(stmt, ir.Assign):
            if isinstance(stmt.target, ir.NameRef):
                clobbers.add(stmt.target)
            else:
                # ensure supported target
                assert isinstance(stmt.target, ir.Subscript)
                for name in walk_parameters(stmt.target):
                    if name not in clobbers:
                        yield name
    elif is_entry_point(stmt):
        # single statement blocks
        if isinstance(stmt, (ir.IfElse, ir.WhileLoop)):
            yield from walk_parameters(stmt.test)
        else:
            assert isinstance(stmt, ir.ForLoop)
            yield from walk_parameters(stmt.iterable)
    elif isinstance(stmt, (ir.SingleExpr, ir.Return)):
        if stmt.value is not None:
            yield from walk_parameters(stmt.value)


def get_read_first_in_block(stmts: List[ir.StmtBase]):
    uevs = set()
    clobbers = set()
    for stmt in stmts:
        uevs.update(get_read_first(stmt, clobbers))
    return uevs


def find_ephemeral_references(node: List[ir.StmtBase], symbols: SymbolTable) -> Set[ir.NameRef]:
    """
    Ephemeral targets are variables that whose assignments can all be localized. This is always decidable in cases
    where a variable is always always bound within a block prior to its first use.

    This looks for ephemeral target references. We consider a reference decidedly ephemeral if it is bound
    first in any block where it is read.
    :return:
    """
    uevs = set()
    for block in depth_first_sequence_blocks(node):
        uevs.update(get_read_first_in_block(block))
    # remove anything is explictly bound and remove everything that
    # is read first in any block.
    bound = set(symbols.assigned_names)
    bound.difference_update(uevs)
    bound.difference_update(symbols.arguments)

    return bound
