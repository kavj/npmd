from collections import defaultdict
from functools import singledispatchmethod

import ir

from visitor import VisitorBase


class UsageCheck(VisitorBase):

    def __call__(self, entry):
        self.entry = entry
        self.used_by = defaultdict(set)
        self.visit(entry)

    @singledispatchmethod
    def get_named_vars(self, node):
        raise TypeError

    @get_named_vars.register
    def _(self, node: ir.NameRef):
        return node

    @get_named_vars.register
    def _(self, node: ir.Expression):
        return {name for name in node.post_order_walk() if isinstance(name, ir.NameRef)}

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Function):
        assert self.entry is node
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.Assign):
        if isinstance(node.target, ir.Expression):
            pass
        self.visit(node.target)
        self.visit(node.value)
        if not isinstance(node.target, ir.Constant):
            self.used_by[node.value].add(node.target)

    @visit.register
    def _(self, node: ir.ForLoop):
        for target, value in node.walk_assignments():
            self.visit(target)
            self.visit(value)
            self.used_by[value].add(target)

