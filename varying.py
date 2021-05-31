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
    """
    This checks for variables that can be privatized

    On scope entry (loop or statement list), we check for upward exposed variables, with the assumption
    that any upward exposed variable is bound along every path that reaches this point.
    An earlier check verifies this, and passes aren't allowed to break that.

    If something is not upward exposed here or following this region, then we privatize the variable.
    The advantage in doing this is that a corresponding variable name can be uniform of varying here, independent
    of the same condition outside this region.

    It's worth noting this isn't a full liveness analysis. It runs early enough that we aren't likely to get
    false positives from other passes, and compound expressions are not yet serialized to three address form and
    therefore cannot create ordering conflicts.

    """

    def __call__(self, entry, uniform_on_entry, declared, assumed_live_out):
        self.entry = entry
        self.varies = {v for v in declared if v not in uniform_on_entry}
        self.uniform = uniform_on_entry.copy()
        self.declared = declared
        use_checker = used_by_check()
        self.used_by, self.assigned_to = use_checker(entry)
        self.visit(entry)
        varies = self.varies
        self.varies = self.uniform = self.used_by = self.assigned_to = self.declared = None
        return varies

    def is_varying(self, node):
        if isinstance(node, ir.NameRef):
            return node in self.varies
        elif isinstance(node, ir.Expression):
            if node in self.varies:
                return True
            return any(subexpr in self.varies for subexpr in node.post_order_walk())
        else:
            raise TypeError

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Constant):
        self.uniform.add(node)

    @visit.register
    def _(self, node: ir.Assign):
        if self.is_varying(node.value):
            self.varies.add(node.target)
            if node.target in self.uniform:
                self.conflicts.add(node.target)

    @visit.register
    def _(self, node: ir.ForLoop):
        for target, iterable in node.assigns:
            if iterable not in self.uniform:
                self.varies.add(target)
        self.visit(node.body)
