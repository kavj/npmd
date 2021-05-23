from functools import singledispatchmethod

import ir
import loopanalysis
import reachingcheck

from visitor import VisitorBase

"""
Convert simple branch bodies to optimized branchless predicate logic, if this is possible.

For example

for i in range(n):
   if a[i] > b[i]:
       a[i] += b[i]

note: unsafe reduction with floating point

# make sure to include a triviality check for cases of all uniform arguments

"""


def find_expr_deps(expr):
    return {e for e in expr.post_order_walk() if isinstance(e, ir.NameRef)}


def exits_from_loop_body(body):
    for stmt in body:
        if isinstance(stmt, (ir.Break, ir.Return)):
            return True
        elif isinstance(stmt, ir.IfElse):
            if exits_from_loop_body(stmt.if_branch) or exits_from_loop_body(stmt.else_branch):
                return True
    return False


def contained_writes(entry):
    """
    return (writes_normal, writes_subscript)

    This way we can determine if it's legal to optimize to identity

    This is useful for a lot of things. For example, we might have a uniform assignment
    in a varying branch, which as a result still requires some kind of predication

    """
    # check separately writes on true branch, writes on false branch
    written_vars = set()
    written_exprs = set()
    for stmt in entry:
        if isinstance(stmt, ir.Assign):
            if isinstance(stmt.target, ir.NameRef):
                written_vars.add(stmt.target)
            else:
                written_exprs.add(stmt.target)
    return written_vars, written_exprs


def predicated_writes(body, varying):
    for stmt in body:
        if isinstance(stmt, ir.IfElse):
            if stmt.test in varying:
                pass


class plan_loop_lowering(VisitorBase):

    def __call__(self, entry, declares, varying, lowering_features):
        self.declares = declares
        self.varying = varying
        self.features = lowering_features
        assert (isinstance(entry, (ir.ForLoop, ir.WhileLoop)))
        self.visit(entry)

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.ForLoop):
        pass
