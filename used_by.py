from collections import defaultdict
from functools import singledispatch, singledispatchmethod

import ir

from visitor import VisitorBase


def get_reads(node):
    if isinstance(node, ir.Expression):
        return {subexpr for subexpr in node.post_order_walk() if isinstance(subexpr, ir.NameRef)}
    elif isinstance(node, ir.NameRef):
        return {node}
    else:
        return set()


@singledispatchmethod
def reads_writes(node):
    raise NotImplementedError


@reads_writes.register
def _(node: ir.Assign):
    if isinstance(node.target, ir.NameRef):
        writes = {node.target}
        reads = get_reads(node.value)
    else:
        writes = set()
        reads = get_reads(node.target).union(get_reads(node.value))
    return reads, writes


@reads_writes.register
def _(node: ir.Expression):
    return {subexpr for subexpr in node.post_order_walk() if not subexpr.constant}, ()


@reads_writes.register
def _(node: ir.ForLoop):
    reads = set()
    writes = set()
    for r, w in node.walk_assignments():
        writes.add(w)
        reads.add(r)
    return reads, writes


@reads_writes.register
def _(node: ir.WhileLoop):
    return get_reads(node.test), set()


@reads_writes.register
def _(node: ir.IfElse):
    return get_reads(node.test), set()


@reads_writes.register
def _(node: ir.SingleExpr):
    return get_reads(node.expr), set()


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
