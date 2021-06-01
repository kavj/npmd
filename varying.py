from collections import defaultdict

import ir
from visitor import walk_all

"""
Very conservative varying checks. These determine uniformity at a variable name level, ignoring local dataflow
information. 

"""


def collect_assigned(entry):
    exprs = defaultdict(set)
    for stmt in walk_all(entry):
        if isinstance(stmt, ir.Assign):
            # only record expressions recorded as top level expressions
            if isinstance(stmt.target, ir.NameRef):
                if not isinstance(stmt.value, ir.Constant):
                    exprs[stmt.value].add(stmt.target)
        elif isinstance(stmt, ir.ForLoop):
            for target, iterable in stmt.walk_assignments():
                exprs[iterable].add(target)
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
