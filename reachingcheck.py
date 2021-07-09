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
            self.mark_assignment(arg, node)
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
        target = node.target
        iterable = node.iterable
        self.mark_reference(iterable)
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


class VarScopeCheck(VisitorBase):
    """
    Checks what variable declarations may be move inside loops.
    This also allows for slightly more aggressive optimization of loop indices.

    The assumption is that if a variable is used, it's bound along all paths reaching that point.
    This condition is checked by ReachingCheck. That allows for a simpler check here.

    """

    def __call__(self, entry):
        self.entry = entry
        self.closed = []
        self.enclosing = set()
        self.bound = set()
        self.visit(entry)
        assert not self.enclosing
        self.closed.append((self.entry, self.bound))
        closed = self.closed
        self.enclosing = self.closed = self.bound = self.entry = None
        return closed

    def unseen(self, node):
        return node not in self.bound and node not in self.enclosing

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.Assign):
        if not isinstance(node.target, ir.Expression) and self.unseen(node.target):
            self.bound.add(node.target)

    @visit.register
    def _(self, node: ir.Function):
        if node is not self.entry:
            # no support for nested scopes
            return
        self.bound.update(node.args)
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.ForLoop):
        prev_bound = self.bound
        if node is not self.entry:
            self.enclosing.update(self.bound)
            self.bound = set()
        target = node.target
        if not isinstance(target, ir.Expression) and self.unseen(target):
            self.bound.add(target)
        self.visit(node.body)
        if node is not self.entry:
            self.closed.append((node, self.bound))
            self.enclosing.difference_update(self.bound)
            self.bound = prev_bound

    @visit.register
    def _(self, node: ir.WhileLoop):
        prev_bound = self.bound
        if node is not self.entry:
            self.enclosing.update(self.bound)
            self.bound = set()
        self.visit(node.body)
        if node is not self.entry:
            self.closed.append((node, self.bound))
            self.enclosing.difference_update(self.bound)
            self.bound = prev_bound


class UsedCheck(VisitorBase):
    """
    Tests what variables are actually read for purposes other than inplace updates
    """

    def __call__(self, entry):
        self.used = set()
        self.visit(entry)
        used = self.used
        self.used = None
        return used

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.Assign):
        if isinstance(node.target, ir.Expression):
            self.visit(node.target)
        self.visit(node.value)

    @visit.register
    def _(self, node: ir.NameRef):
        self.used.add(node)

    @visit.register
    def _(self, node: ir.ForLoop):
        target = node.target
        value = node.iterable
        if isinstance(target, ir.Expression):
            self.used.add(target)
        self.used.add(value)
