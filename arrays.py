from functools import singledispatchmethod

import ir
from visitor import TransformBase


class ArrayRefRepl(TransformBase):

    def __init__(self):
        self.types = None

    def replace_array_refs(self, entry, types):
        self.types = types
        assert (isinstance(entry, ir.Function))
        self.visit(entry)

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Function):
        body = [self.visit(n) for n in node.body]
        node.body = body

    @visit.register
    def _(self, node: ir.IfElse):
        node.test = self.visit(node.test)
        node.if_branch = self.visit(node.if_branch)
        node.else_branch = self.visit(node.else_branch)
        return node

    @visit.register
    def _(self, node: ir.ForLoop):
        node.iterable = self.visit(node.iterable)
        node.body = self.visit(node.body)

    @visit.register
    def _(self, node: ir.WhileLoop):
        node.test = self.visit(node.test)
        node.body = self.visit(node.body)

    @visit.register
    def _(self, node: ir.Assign):
        self.visit(node.target)
        self.visit(node.value)

    @visit.register
    def _(self, node: ir.CascadeAssign):
        node.value = self.visit(node.value)
        node.targets = [self.visit(t) for t in node.targets]

    @visit.register
    def _(self, node):
        pass

    @visit.register
    def _(self, node: ir.ArrayRef):
        ndims = self.visit(node.ndims)
        dtype = self.visit(node.dtype)
        base = self.visit(node.base) if node.base is not None else None
        return ir.ArrayRef(ndims, dtype, base)

    @visit.register
    def _(self, node: ir.ShapeRef):
        array = self.visit(node.array)
        return ir.ShapeRef(array)
