import numbers

import ir
from errors import CompilerError


def get_expr_parameters(expr):
    return {subexpr for subexpr in walk(expr) if isinstance(subexpr, ir.NameRef)}


def wrap_constant(c):
    if isinstance(c, bool):
        return ir.BoolConst(c)
    if isinstance(c, numbers.Integral):
        return ir.IntConst(c)
    elif isinstance(c, numbers.Real):
        return ir.FloatConst(c)
    else:
        msg = f"Can't construct constant node for unsupported constant type {type(c)}"
        raise NotImplementedError(msg)


def unpack_assignment(target, value, pos):
    if isinstance(target, ir.Tuple) and isinstance(value, ir.Tuple):
        if target.length != value.length:
            msg = f"Cannot unpack {value} with {value.length} elements using {target} with {target.length} elements: " \
                  f"line {pos.line_begin}."
            raise ValueError(msg)
        for t, v in zip(target.subexprs, value.subexprs):
            yield from unpack_assignment(t, v, pos)
    else:
        yield target, value


def unpack_iterated(target, iterable, pos):
    if isinstance(iterable, ir.Zip):
        # must unpack
        if isinstance(target, ir.Tuple):
            if len(target.elements) == len(iterable.elements):
                for t, v in zip(target.elements, iterable.elements):
                    yield from unpack_iterated(t, v, pos)
            else:
                msg = f"Mismatched unpacking counts for {target} and {iterable}, {len(target.elements)} " \
                      f"and {(len(iterable.elements))}."
                raise CompilerError(msg)
        else:
            msg = f"Zip construct {iterable} requires a tuple for unpacking."
            raise CompilerError(msg)
    elif isinstance(iterable, ir.Enumerate):
        if isinstance(target, ir.Tuple):
            if len(target.elements) == 2:
                first_target, sec_target = target.elements
                yield first_target, iterable.iterable
                yield sec_target, iterable.start
            else:
                msg = f"Enumerate must be unpacked to exactly two targets, received {len(target.elements)}."
                raise CompilerError(msg)
        else:
            msg = f"Only tuples are supported unpacking targets. Received {type(target)}."
            raise CompilerError(msg)
    else:
        # Array or sequence reference, with a single opaque target.
        yield target, iterable


def is_pow(expr):
    return isinstance(expr, ir.BinOp) and expr.op in ("**", "**=")


def is_fma_pattern(expr):
    """
    This ignores safety issues, which may be addressed later for anything
    that looks like a * b - c * d.


    """

    if isinstance(expr, ir.BinOp) and expr.op in ("+", "+=", "-", "-="):
        left = expr.left
        right = expr.right
        for operand in (left, right):
            if isinstance(operand, ir.BinOp):
                if operand.op == "*":
                    return True
            elif isinstance(operand, ir.UnaryOp):
                # Expression simplifiers should have already folded any multiple
                # nestings of unary -
                if (operand.op == "-"
                        and isinstance(operand.operand, ir.BinOp)
                        and operand.operand.op == "*"):
                    return True
    return False


def is_addition(node):
    return isinstance(node, ir.BinOp) and node.op in ("+", "+=")


def is_subtraction(node):
    return isinstance(node, ir.BinOp) and node.op in ("-", "-=")


def is_multiplication(node):
    return isinstance(node, ir.BinOp) and node.op in ("*", "*=")


def is_division(node):
    return isinstance(node, ir.BinOp) and node.op in ("/", "//", "/=", "//=")


def equals_unary_negate(node):
    return isinstance(node, ir.UnaryOp) and node.op == "-"
