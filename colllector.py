from collections import defaultdict
from functools import singledispatchmethod

import ir

from visitor import VisitorBase


class Collector(VisitorBase):
    """
    This maps assigned targets to anything assigned to them,
    expressions to their named dependencies,
    and names to expressions where they are used.

    It's useful for determining what varies across consecutive interleaved calls.

    """

    def __call__(self, func):
        self.by_target = defaultdict(set)
        self.by_expr = {}
        self.visit(func.body)
        by_target = self.by_target
        by_expr = self.by_expr
        expr_by_var = self.expr_by_var
        self.by_target = self.by_expr = self.expr_by_var = None
        return by_target, by_expr, expr_by_var

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Assign):
        self.visit(node.target)
        self.by_target[node.target].add(node.value)
        self.visit(node.value)

    @visit.register
    def _(self, node: ir.Expression):
        if node not in self.by_expr:
            deps = set()
            for subexpr in node.subexprs:
                if isinstance(subexpr, ir.NameRef):
                    deps.add(subexpr)
                elif isinstance(subexpr, ir.Expression):
                    self.visit(subexpr)
                    deps.update(self.by_expr[subexpr])

    @visit.register
    def _(self, node: ir.ForLoop):
        for target, value in node.walk_assignments():
            self.visit(target)
            self.visit(value)
            self.by_target[target].add(value)
