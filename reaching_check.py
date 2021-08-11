from contextlib import contextmanager
from functools import singledispatchmethod

import ir
from lowering import unpack_iterated
from visitor import StmtVisitor, walk


def get_expr_parameters(expr):
    return {subexpr for subexpr in walk(expr) if isinstance(subexpr, ir.NameRef)}


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

    def register_reference(self, ref, stmt):
        for param in get_expr_parameters(ref):
            if param not in self.unknowns:
                if self.maybe_unbound(param):
                    self.unknowns[param] = stmt
        for scope in reversed(self._bound):
            if ref in scope:
                return True
        return False

    def register_assignment(self, target, value, stmt):
        self.register_reference(value, stmt)
        self.register_reference(target, stmt)
        self.mark_assigned(target)

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
        self.register_reference(node.expr, node)

    @visit.register
    def _(self, node: ir.Assign):
        self.register_assignment(node.target, node.value, node)

    @visit.register
    def _(self, node: ir.ForLoop):
        with self.scoped():
            for target, value in unpack_iterated(node.target, node.iterable, node.pos):
                self.register_assignment(target, value, node)
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.WhileLoop):
        # test is encountered before loop
        self.register_assignment(None, node.test, node)
        with self.scoped():
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.IfElse):
        self.register_reference(node.test, node)
        with self.scoped():
            self.visit(node.if_branch)
            if_branch = self.current_scope
        with self.scoped():
            self.visit(node.else_branch)
            else_branch = self.current_scope
        definitely_bound = if_branch.intersection(else_branch)
        self.current_scope.update(definitely_bound)
