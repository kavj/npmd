import operator

import ir
from visitor import walk_multiple


# anything that doesn't fit elsewhere at the moment


def unpack_expressions(exprs):
    """
    Expands all expressions, yielding

    top: a set of expressions, which are not sub-expressions of any other
    params: the free variable parameters required by each expression
    expr_set: the set of all expressions, expanded to include sub-expressions

    """
    params = {}
    subexprs = set()
    expr_set = set()
    for expr in walk_multiple(exprs):
        if not expr.is_expr:
            continue
        expr_set.add(expr)
        p = set()
        for subexpr in expr.subexprs:
            if subexpr.is_expr:
                subexprs.add(subexpr)
                p.update(params.get(subexpr))
            else:
                p.add(subexpr)
        params[expr] = p
    top = expr_set.difference(subexprs)
    expr_set.difference_update(top)
    top = {t: params[t] for t in top}
    subexprs = {s: params[s] for s in expr_set}
    return top, subexprs


def map_permute(lhs: ir.Tuple, rhs: ir.Tuple):
    """
    Maps unpacking of a permutation or short array of constant length

    """

    # Check for duplicate target assignments, like:  b, a, a = a, b, c
    # CPython won't raise an error here, but we do.

    assigned = {}
    for t, v in zip(lhs.elements, rhs.elements):
        if t in assigned:
            raise KeyError("duplicate assignment in permutation")
        assigned[t] = v

    return assigned


def negate_condition(node):
    repl = ir.UnaryOp(node, "not")
    if node.is_constant:
        repl = ir.BoolNode(operator.invert(operator.truth(node)))
    elif isinstance(node, ir.UnaryOp):
        if node.op == "not":
            repl = node.operand
    elif isinstance(node, ir.CompareOp):
        # Only fold cases with a single operator that has a negated form.
        # Otherwise we have to worry about edge cases involving unordered operands.
        if len(node.operands) == 2 and len(node.ops) == 1:
            if node.ops[0] == "==":
                repl = ir.CompareOp(node.operands, ("!=",))
            elif node.ops[0] == "!=":
                repl = ir.CompareOp(node.operands, ("==",))
    return repl


def contains_subscript(node):
    if node.is_subscript:
        return True
    elif node.is_expr:
        queued = [node.subexprs]
        while queued:
            try:
                d = next(queued[-1])
                if d.is_subscript:
                    return True
                elif hasattr(d, 'subexprs'):
                    queued.append(d.subexprs)
            except StopIteration:
                pass
    return False
