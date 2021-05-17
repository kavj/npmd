from functools import singledispatchmethod

import ir
from visitor import VisitorBase


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

    def may_be_unbound(self, value):
        if value.constant:
            return False
        if value in self.seen:
            return False
        for scope in self.enclosing:
            if value in scope:
                return False
        return True

    def mark_reference(self, node, stmt=None):
        if isinstance(node, ir.Expression):
            for e in node.subexprs:
                self.mark_reference(e, stmt)
        elif self.may_be_unbound(node):
            if node not in self.unknowns:
                self.unknowns[node] = stmt
        else:
            self.raw.add(node)

    def mark_assignment(self, target, stmt):
        if isinstance(target, ir.Expression):
            self.mark_reference(target, stmt)
        else:
            if target in self.seen:
                self.war.add(target)
            self.seen.add(target)

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: list):
        repl = []
        for stmt in node:
            repl.append(self.visit(stmt))
        return repl

    @visit.register
    def _(self, node: ir.Module):
        raise NotImplementedError("Reaching pass is meant to run on a per function basis.")

    @visit.register
    def _(self, node: ir.Function):
        # no support for nested scopes
        assert (node is self.entry)
        for arg in node.args:
            self.mark_assignment(arg.name, node)
        for stmt in node.body:
            self.visit(stmt)

    @visit.register
    def _(self, node: ir.SingleExpr):
        self.mark_reference(node.expr, node)

    @visit.register
    def _(self, node: ir.NameRef):
        self.mark_reference(node)

    @visit.register
    def _(self, node: ir.Expression):
        self.mark_reference(node)

    @visit.register
    def _(self, node: ir.ShapeRef):
        # not set up as an expression right now... instantiate so it stops coming up as unbound
        return

    @visit.register
    def _(self, node: ir.Assign):
        if node.in_place:
            self.mark_reference(node.target, node)
            self.mark_reference(node.value, node)
        else:
            self.mark_reference(node.value, node)
            self.mark_assignment(node.target, node.value)

    @visit.register
    def _(self, node: ir.ForLoop):
        seen = self.seen
        self.enclosing.append(seen)
        self.seen = set()
        # mark iterables first
        for _, value in node.assigns:
            self.mark_reference(value)
        for target, _ in node.assigns:
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
        self.seen = set()
        self.visit(node.if_branch)
        seen_if = self.seen
        self.seen = set()
        self.visit(node.else_branch)
        seen_else = self.seen
        self.seen = seen
        seen = self.enclosing.pop()
        assert (seen is self.seen)
        seen_ifelse = seen_if.intersection(seen_else)
        self.seen.update(seen_ifelse)
