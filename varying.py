from collections import defaultdict
from functools import singledispatchmethod

import ir
import visitor

"""
Determines whether things should be assumed to vary across calls

The motivation here is that we may have an array like calculation, with something like

for row in ndarray:
    output[i] = f(row)
    
or a sliding window type..

for i in range(n):
    # window along leading dim
    output[i] = f(ndarray[i:i+windowlen])

Within the call itself, we may have a separate

for i in ...:
   for j in ...:
       for k in ...:
           ...
           
It's possible for the intervals of i,j,k to be uniform across calls, in which case
we only need to wrap varying data in the innermost loop.

This is valuable in that it allows function level vectorization to be converted to inner loop vectorization
in cases with varying dataflow and uniform control flow.


"""


class used_by_check(visitor.VisitorBase):

    def __call__(self, entry):
        self.used_by = defaultdict(set)
        self.assigned_to = defaultdict(set)
        self.visit(entry)
        used_by = self.used_by
        assigned_to = self.assigned_to
        self.used_by = self.assigned_to = None
        return used_by, assigned_to

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.Expression):
        for subexpr in node.subexprs:
            self.visit(subexpr)
            self.used_by[subexpr].add(node)

    @visit.register
    def _(self, node: ir.MinConstraint):
        # exception, as these can still have uniform length parameters
        # pass should be run before these are introduced anyway
        pass

    @visit.register
    def _(self, node: ir.Assign):
        self.visit(node.target)
        self.visit(node.value)
        self.assigned_to[node.target].add(node.value)


class varying_check(visitor.VisitorBase):

    def __call__(self, entry, uniform_args, uniform_len_args):
        assert isinstance(entry, ir.Function)
        self.varies = {v for v in entry.args if v not in uniform_args}
        self.uniform = uniform_args.copy()
        self.uniform_len_args = uniform_len_args
        self.used_by = set()
        self.visit(entry)
        varies = self.varies
        self.varies = self.uniform = self.uniform_len_args = None
        return varies

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Constant):
        self.uniform.add(node)

    @visit.register
    def _(self, node: ir.Expression):
        for subexpr in node.subexprs:
            self.visit(subexpr)
        if all(subexpr in self.uniform for subexpr in node.subexprs):
            self.uniform.add(node)
        else:
            self.varies.add(node)

    @visit.register
    def _(self, node: ir.ForLoop):
        for target, iterable in node.assigns:
            if iterable not in self.uniform:
                self.varies.add(target)
        self.visit(node.body)
