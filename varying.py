from collections import defaultdict
from functools import singledispatchmethod

import ir
from visitor import StmtVisitor, walk

"""
Very conservative varying checks. These determine uniformity at a variable name level, ignoring local dataflow
information. 

"""


def if_convert_branch(node: ir.IfElse):
    """
    Convert branch to a sequence of ternary ops where possible
    """
    pass


class MarkVaryingVisitor(StmtVisitor):

    def __init__(self):
        self.varying = None

    def __call__(self, entry, varying):
        self.varying = varying
        self.visit(entry)

    @singledispatchmethod
    def visit(self, node):
        pass


class VaryingApproximate(StmtVisitor):

    def __init__(self):
        self.varying = None

    def __call__(self, entry, varying_args):
        self.varying = varying_args.copy()

    @singledispatchmethod
    def is_varying(self, node):
        raise NotImplementedError

    @is_varying.register
    def _(self, node: ir.Assign):
        pass

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Assign):
        pass


class MapDependentExprs(StmtVisitor):

    def __init__(self):
        self.deps = None

    def __call__(self, entry):
        self.deps = {}
        self.visit(entry)
        deps = self.deps
        self.deps = None
        return deps

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.ValueRef):
        expr_deps = set()
        for subexpr in node.subexprs:
            self.visit(subexpr)
            expr_deps.add(subexpr)
        self.deps[node] = expr_deps


class AssignmentRef:

    def __init__(self, target, value, iterated=False):
        self.target = target
        self.value = value
        self.iterated = iterated


class MapTargets(StmtVisitor):

    def __init__(self):
        self.targets = None

    def __call__(self, entry):
        self.targets = defaultdict(set)
        self.visit(entry)
        targets = self.targets
        self.targets = None
        return targets

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Assign):
        target = node.target
        self.targets[target].add(AssignmentRef(target, node.value))

    @visit.register
    def _(self, node: ir.ForLoop):
        raise NotImplementedError("UNDER RECONSTRUCTION")


def collect_assigned(entry):
    exprs = defaultdict(set)
    for stmt in walk(entry):
        if isinstance(stmt, ir.Assign):
            # only record expressions recorded as top level expressions
            if isinstance(stmt.target, ir.NameRef):
                if not stmt.value.constant:
                    exprs[stmt.value].add(stmt.target)
        elif isinstance(stmt, ir.ForLoop):
            raise NotImplementedError("UNDER RECONSTRCTION")

    return exprs


def map_parameters(exprs):
    params = {}
    for expr in exprs:
        p = {subexpr for subexpr in expr.post_order_walk() if isinstance(subexpr, ir.NameRef)}
        params[expr] = p
    return params


def varying_from_exprs(exprs, params, varying_inputs):
    varying = varying_inputs.copy()
    changed = True
    while changed:
        changed = False
        # update expressions
        for expr, targets in exprs.items():
            if expr in varying:
                continue
            if any(p in varying for p in params.get(expr, ())):
                changed = True
                varying.add(expr)
                for target in targets:
                    varying.add(target)
    return varying


def find_varying(entry, varying_inputs):
    exprs = collect_assigned(entry)
    params = map_parameters(exprs.keys())
    varying = varying_from_exprs(exprs, params, varying_inputs)
    return varying
