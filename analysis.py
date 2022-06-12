import operator

import numpy as np

from collections import defaultdict
from contextlib import contextmanager
from functools import singledispatchmethod
from typing import Iterable, List, Optional, Set

import ir

from folding import OpFold
from symbol_table import SymbolTable
from type_checks import TypeHelper
from utils import is_entry_point, unpack_iterated
from traversal import depth_first_sequence_blocks, StmtVisitor, walk_parameters


# Todo: stub
specialized = {ir.NameRef("print")}


def has_nan(node: ir.Expression):
    for subexpr in node.subexprs:
        if isinstance(subexpr, ir.CONSTANT):
            if np.isnan(subexpr.value):
                return True


def is_compare(node):
    """
    This tests whether we have either a single compare or a chained comparison.
    Chained comparisons are regarded as logical and nodes, where every operand
    is a compare operation and for any consecutive comparisons, indexed by 'i' and 'i+1':
        operands[i].right == operands[i+1].left

    Note, this will ignore cases that are not laid out like:

        (a cmp b) and (b cmp c) and (c cmp d)

    but the internal passes attempt to factor everything this way.

    """

    if isinstance(node, ir.CompareOp):
        return True

    elif isinstance(node, ir.AND):
        operand_iter = node.subexprs
        first = next(operand_iter)
        if not isinstance(first, ir.CompareOp):
            return False
        prev_rhs = first.right
        for operand in operand_iter:
            if not isinstance(operand, ir.CompareOp) or prev_rhs != operand.left:
                return False

    else:
        return False

    return True


def get_possible_aliasing(stmts: Iterable[ir.StmtBase], symbols: SymbolTable):
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


class ReachingCheck(StmtVisitor):
    """
    This is meant to check for statements that could result in unbound local errors.
    It also tracks cases where a write must follow a read or a read follows a write.

    """

    def __init__(self):
        self.unknowns = None
        self.specialized = specialized.copy()
        self._bound = []

    def __call__(self, node):
        # Needs to have context added so we can check imported symbols
        self.unknowns = {}
        assert not self._bound
        self.visit(node)
        assert not self._bound
        unknowns = self.unknowns
        self.unknowns = None
        return unknowns

    @contextmanager
    def scoped(self):
        innermost = set()
        self._bound.append(innermost)
        yield
        p = self._bound.pop()
        assert p is innermost

    @property
    def current_scope(self):
        return self._bound[-1]

    def mark_assigned(self, target):
        if isinstance(target, ir.NameRef):
            self.current_scope.add(target)

    def maybe_unbound(self, name):
        if name in self.specialized:
            return False
        for scope in reversed(self._bound):
            if name in scope:
                return False
        return True

    def register_name_reference(self, name, stmt):
        if name not in self.unknowns:
            if self.maybe_unbound(name):
                self.unknowns[name] = stmt

    def register_expr_reference(self, expr, stmt):
        for param in walk_parameters(expr):
            self.register_name_reference(param, stmt)
        if isinstance(expr, ir.NameRef):
            self.register_name_reference(expr, stmt)

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.Function):
        with self.scoped():
            for arg in node.args:
                self.mark_assigned(arg)
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.SingleExpr):
        self.register_expr_reference(node.value, node)

    @visit.register
    def _(self, node: ir.InPlaceOp):
        self.register_expr_reference(node.value, node)

    @visit.register
    def _(self, node: ir.Assign):
        self.register_expr_reference(node.value, node)
        self.mark_assigned(node.target)
        # catch non-name targets
        self.register_expr_reference(node.target, node)

    @visit.register
    def _(self, node: ir.ForLoop):
        with self.scoped():
            for target, value in unpack_iterated(node.target, node.iterable):
                self.register_expr_reference(value, node)
                self.mark_assigned(target)
                self.register_expr_reference(target, node)
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.WhileLoop):
        # test is encountered before loop
        self.register_expr_reference(node.test, node)
        with self.scoped():
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.IfElse):
        self.register_expr_reference(node.test, node)
        with self.scoped():
            self.visit(node.if_branch)
            if_branch = self.current_scope
        with self.scoped():
            self.visit(node.else_branch)
            else_branch = self.current_scope
        definitely_bound = if_branch.intersection(else_branch)
        # If bound in both branches, mark as bound.
        # Declarations must be hoisted if these may
        # escape.
        self.current_scope.update(definitely_bound)

    @visit.register
    def _(self, node: ir.Return):
        if node.value is not None:
            self.register_expr_reference(node.value, node)


class AllPathsReturn(StmtVisitor):
    """
    This needs to check return type

    """

    def __call__(self, node):
        terminated = self.visit(node)
        return terminated

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.Function):
        return self.visit(node.body)

    @visit.register
    def _(self, node: ir.StmtBase):
        return False

    @visit.register
    def _(self, node: ir.ForLoop):
        return False

    @visit.register
    def _(self, node: ir.WhileLoop):
        # ignoring the case of while True: .. return ..
        # which basically works if return is the only way out
        return False

    @visit.register
    def _(self, node: ir.Return):
        return True

    @visit.register
    def _(self, node: list):
        for stmt in reversed(node):
            if self.visit(stmt):
                return True
        return False

    @visit.register
    def _(self, node: ir.IfElse):
        if isinstance(node.test, ir.CONSTANT):
            if operator.truth(node.test):
                return self.visit(node.if_branch)
            else:
                return self.visit(node.else_branch)
        else:
            return self.visit(node.if_branch) and self.visit(node.else_branch)


def find_unterminated_path(stmts: List[ir.StmtBase]) -> Optional[List[ir.StmtBase]]:
    if not isinstance(stmts, list):
        raise TypeError("Internal Error: expected a list of statements")
    if len(stmts) > 0:
        last = stmts[-1]
        if isinstance(last, (ir.Continue, ir.Break, ir.Return)):
            return
        elif isinstance(last, ir.IfElse):
            if last.test.constant:
                # If we have a constant branch condition,we can only follow
                # the reachable branch
                if operator.truth(last.test):
                    return find_unterminated_path(last.if_branch)
                else:
                    return find_unterminated_path(last.else_branch)
            else:
                if_path = find_unterminated_path(last.if_branch)
                else_path = find_unterminated_path(last.else_branch)
                if if_path is None and else_path is None:
                    return  # terminated
                elif if_path is None:
                    return else_path
                elif else_path is None:
                    return if_path
    return stmts


def compute_element_count(start: ir.ValueRef, stop: ir.ValueRef, step: ir.ValueRef, typer: TypeHelper):
    """
    Single method to compute the number of elements for an iterable expression.

    :param start:
    :param stop:
    :param step:
    :return:
    """
    fold_op = OpFold(typer)

    start = fold_op(start)
    stop = fold_op(stop)
    step = fold_op(step)
    diff = fold_op(ir.SUB(stop, start))
    base_count = fold_op((ir.FLOORDIV(diff, step)))
    test = fold_op(ir.MOD(diff, step))
    fringe = fold_op(ir.SELECT(predicate=test, on_true=ir.One, on_false=ir.Zero))
    count = fold_op(ir.ADD(base_count, fringe))
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
    fold_op = OpFold(typer)
    if isinstance(node, ir.Subscript):
        if isinstance(node.index, ir.Slice):
            index = node.index
            start = index.start
            if index.stop is None:
                stop = ir.SingleDimRef(node.value, dim=ir.Zero)
            else:
                stop = fold_op(ir.MIN(index.stop, ir.SingleDimRef(node.value, dim=ir.Zero)))
            step = index.step
            return compute_element_count(start, stop, step, typer)
        else:
            # first dim removed
            return ir.SingleDimRef(node.value, dim=ir.One)
    elif isinstance(node, ir.NameRef):
        return ir.SingleDimRef(node, dim=ir.Zero)
    # other cases handled by analyzing sub-expressions and broadcasting constraints


def get_uevs(stmt: ir.StmtBase, clobbers: Set[ir.NameRef]):
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


def find_ephemeral_references(node: Iterable[ir.StmtBase], symbols: SymbolTable) -> Set[ir.NameRef]:
    """
    Ephemeral targets are variables that whose assignments can all be localized. This is always decidable in cases
    where a variable is always always bound within a block prior to its first use.

    This looks for ephemeral target references. We consider a reference decidedly ephemeral if it is bound
    first in any block where it is read.
    :return:
    """
    uevs = set()
    clobbers = set()
    for block in depth_first_sequence_blocks(node):
        clobbers.clear()
        for stmt in block:
            uevs.update(get_uevs(stmt, clobbers))
    # remove anything is explictly bound and remove everything that
    # is read first in any block.
    bound = set(symbols.assigned_names)
    bound.difference_update(uevs)
    bound = {b for b in bound if not symbols.is_argument(b)}
    return bound
