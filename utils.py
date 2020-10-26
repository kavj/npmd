import operator

import ir


# anything that doesn't fit elsewhere at the moment


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
