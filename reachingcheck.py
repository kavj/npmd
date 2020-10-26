from functools import singledispatchmethod

import ir
from visitor import VisitorBase, walk_assignments


class ReachingCheck(VisitorBase):
    """
    This is meant to check for statments that could result in unbound local errors. 
    It also tracks cases where a write must follow a read or a read follows a write.

    """

    def __call__(self, node):
        # Needs to have context added so we can check imported symbols
        self.enclosing = []
        self.entry = node
        self.unknowns = {}
        self.seen = set()
        self.raw = set()
        self.war = set()
        self.visit(node)
        assert (not self.enclosing)
        seen, unknowns, raw, war = self.seen, self.unknowns, self.raw, self.war
        self.seen = self.unknowns = self.raw = self.war = self.enclosing = None
        return unknowns, raw, war

    def may_be_unbound(self, value: ir.ObjectBase):
        if value.is_constant:
            return False
        if value in self.seen:
            return False
        for scope in self.enclosing:
            if value in scope:
                return False
        return True

    def mark_reference(self, node, stmt=None):
        if node.is_expr:
            for e in node.subexprs:
                self.mark_reference(e, stmt)
        elif self.may_be_unbound(node):
            if node not in self.unknowns:
                self.unknowns[node] = stmt
        else:
            self.raw.add(node)

    def mark_assignment(self, target, stmt):
        if target.is_expr:
            self.mark_reference(target, stmt)
        else:
            if target in self.seen:
                self.war.add(target)
            self.seen.add(target)

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.Module):
        raise NotImplementedError("Reaching pass is meant to run on a per function basis.")

    @visit.register
    def _(self, node: ir.Function):
        # no support for nested scopes
        assert (node is self.entry)
        for arg in node.args:
            self.mark_assignment(arg.name, node)
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.SingleExpr):
        self.mark_reference(node.expr, node)

    @visit.register
    def _(self, node: ir.ObjectBase):
        self.mark_reference(node)

    @visit.register
    def _(self, node: ir.Assign):
        if node.in_place:
            self.mark_reference(node.target, node)
            self.mark_reference(node.value, node)
        else:
            for _, value in walk_assignments(node):
                self.mark_reference(value, node)
            for target, value in walk_assignments(node):
                self.mark_assignment(target, value)

    @visit.register
    def _(self, node: ir.CascadeAssign):
        self.mark_reference(node.value, node)
        for t in node.targets:
            self.mark_assignment(t, node)

    @visit.register
    def _(self, node: ir.ForLoop):
        seen = self.seen
        self.enclosing.append(seen)
        self.seen = set()
        # mark iterables first
        for _, iterable in walk_assignments(node):
            self.mark_reference(iterable, node)
            # Skip assignments that are assumed to be
            # unpacking mismatches.
        for target, iterable in walk_assignments(node):
            self.mark_assignment(target, node)
        self.visit(node.body)
        self.seen = self.enclosing.pop()
        assert (seen is self.seen)

    @visit.register
    def _(self, node: ir.WhileLoop):
        seen = self.seen
        self.enclosing.append(seen)
        self.seen = set()
        self.mark_reference(node.test, node)
        self.visit(node.body)
        self.seen = self.enclosing.pop()
        assert (seen is self.seen)

    @visit.register
    def _(self, node: ir.IfElse):
        self.visit(node.test)
        seen = self.seen
        self.enclosing.append(seen)
        seen_if_branch = set()
        seen_else_branch = set()
        self.seen = seen_if_branch
        self.visit(node.if_branch)
        self.seen = seen_else_branch
        self.visit(node.else_branch)
        self.seen = self.enclosing.pop()
        assert (seen is self.seen)
        self.seen.update(seen_if_branch.intersection(seen_else_branch))
