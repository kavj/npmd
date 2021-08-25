from contextlib import contextmanager
from functools import singledispatchmethod

import ir
from utils import unpack_iterated, get_expr_parameters
from visitor import StmtVisitor


class ReachingCheck(StmtVisitor):
    """
    This is meant to check for statements that could result in unbound local errors.
    It also tracks cases where a write must follow a read or a read follows a write.

    """

    def __init__(self):
        self.unknowns = None
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
        for scope in reversed(self._bound):
            if name in scope:
                return False
        return True

    def register_name_reference(self, name, stmt):
        if name not in self.unknowns:
            if self.maybe_unbound(name):
                self.unknowns[name] = stmt

    def register_expr_reference(self, expr, stmt):
        for param in get_expr_parameters(expr):
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
        self.register_expr_reference(node.expr, node)

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