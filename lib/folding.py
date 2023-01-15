import itertools

from functools import singledispatch
from typing import Iterable, Iterator, Union

import numpy as np

import lib.ir as ir

from lib.errors import CompilerError
from lib.type_checks import is_integer, TypeHelper


@singledispatch
def simplify_untyped_numeric(expr):
    msg = f'"{expr}" is not a visitable value reference or expression.'
    raise TypeError(msg)


@simplify_untyped_numeric.register
def _(expr: ir.ValueRef):
    return expr


@simplify_untyped_numeric.register
def _(expr: ir.ADD):
    # if we have -a + b, then convert to b - a, since this sanitizes some undefined behavior
    for u,v in itertools.permutations(expr.subexprs):
        if isinstance(v, ir.USUB) and not isinstance(u, ir.USUB):
            return ir.SUB(u, v.operand)
    return expr


@simplify_untyped_numeric.register
def _(expr: ir.SUB):
    # If we have -a + b
    u, v = expr.subexprs
    if isinstance(v, ir.USUB):
        return ir.ADD(u, v.operand)
    return expr


@simplify_untyped_numeric.register
def _(expr: ir.USUB):
    if isinstance(expr.operand, ir.USUB):
        return expr.operand.operand
    return expr


@simplify_untyped_numeric.register
def _(expr: ir.MULT):
    for u, v in itertools.permutations(expr.subexprs):
        if isinstance(u, ir.CONSTANT) and u.is_integer:
            if u == ir.One:
                return v
            elif u == ir.Two:
                repl = ir.ADD(v, v)
                return repl
    return expr


@simplify_untyped_numeric.register
def _(expr: ir.TRUEDIV):
    if expr.right == ir.Two:
        return ir.MULT(ir.Half, expr.left)
    return expr


@simplify_untyped_numeric.register
def _(expr: ir.FLOORDIV):
    # fold if integer one only
    if isinstance(expr.right, ir.CONSTANT) and expr.right == ir.One and expr.right.is_integer:
        return ir.FLOOR(expr.left)


@simplify_untyped_numeric.register
def _(expr: ir.POW):
    if not isinstance(expr.right, ir.CONSTANT):
        return expr
    # simplify up to power of 4
    if expr.right == ir.Half:
        return ir.SQRT(expr.left)
    elif expr.right.negative:
        right = ir.wrap_constant(-expr.right.value)
        denominator = ir.POW(expr.left, right)
        repl_denominator = simplify_untyped_numeric(denominator)
        return ir.TRUEDIV(ir.One, repl_denominator)
    elif expr.right.is_integer:
        if 0 < expr.right.value <= 4:
            # don't simplify to zero is to avoid inferring type
            if expr.right == ir.One:
                return expr.left
            elif expr.right == ir.Two:
                return ir.MULT(expr.left, expr.left)
            elif expr.right == ir.wrap_constant(3):
                subexpr = ir.MULT(expr.left, expr.left)
                return ir.MULT(subexpr, expr.left)
            else:
                subexpr = ir.MULT(expr.left, expr.left)
                return ir.MULT(subexpr, subexpr)
    else:
        return expr


@simplify_untyped_numeric.register
def _(expr: ir.CAST):
    if isinstance(expr.value, ir.CAST):
        expr = ir.CAST(expr.value.value, expr.target_type)
    return expr
