import itertools
import math
import operator

from functools import singledispatch
from typing import Iterable, Iterator, Union

import numpy as np

import npmd.ir as ir

from npmd.errors import CompilerError
from npmd.type_checks import is_integer, TypeHelper


def concatenate(args: Union[Iterable[ir.Expression], Iterator]):
    """
    Method to conjoin two or more arguments as a tuple
    :param args:
    :return:
    """
    terms = []
    for arg in args:
        if isinstance(arg, ir.TUPLE):
            terms.extend(arg.elements)
        else:
            terms.append(arg)
    if len(terms) < 2:
        msg = f'Concatenate requires at least 2 valid items, received "{terms}"'
        raise ValueError(msg)
    return ir.TUPLE(*terms)


@singledispatch
def fold_constants(node):
    return node


@fold_constants.register
def _(node: ir.BinOp):
    if isinstance(node.left, ir.CONSTANT) and isinstance(node.right, ir.CONSTANT):
        op = const_ops[type(node)]
        return ir.wrap_constant(op(node.left.value, node.right.value))
    return node


@fold_constants.register
def _(node: ir.MAX):
    left = node.left
    right = node.right
    if isinstance(left, ir.CONSTANT) and isinstance(right, ir.CONSTANT):
        if left.is_bool or right.is_bool:
            # bad idea to mess with this
            return node
        elif left.is_nan:
            return right
        elif right.is_nan:
            return left
        else:
            dtype = np.promote_types(left.dtype, right.dtype)
            # cast nodes
            left = dtype.type(left.value)
            right = dtype.type(right.value)
            result = np.max((left, right))
            return ir.wrap_constant(result)
    return node


@fold_constants.register
def _(node: ir.MIN):
    left = node.left
    right = node.right
    if isinstance(left, ir.CONSTANT) and isinstance(right, ir.CONSTANT):
        if left.is_bool or right.is_bool:
            # bad idea to mess with this
            return node
        elif left.is_nan:
            return right
        elif right.is_nan:
            return left
        else:
            dtype = np.promote_types(left.dtype, right.dtype)
            # cast nodes
            left = dtype.type(left.value)
            right = dtype.type(right.value)
            result = np.min((left, right))
            return ir.wrap_constant(result)
    return node


@fold_constants.register
def _(node: ir.CompareOp):
    if isinstance(node.left, ir.CONSTANT) and isinstance(node.right, ir.CONSTANT):
        op = const_ops[type(node)]
        return ir.wrap_constant(op(node.left.value, node.right.value))
    return node


@fold_constants.register
def _(node: ir.IN):
    return node


@fold_constants.register
def _(node: ir.NOTIN):
    return node


class ExprContract:
    """
    Try to form an expression of the form

    a * b + c
    -(a * b) + c
    a * b - c

    not fusable unless it's a floating point op

    from the given expression
    """
    float_dtypes = (np.dtype('float32'), np.dtype('float64'))

    def is_float(self, expr: ir.ValueRef) -> bool:
        t = self.typer(expr)
        if isinstance(t, ir.ArrayType):
            return t.dtype in self.float_dtypes
        return t in self.float_dtypes

    def __init__(self, typer: TypeHelper):
        self.typer = typer

    def __call__(self, expr: ir.ValueRef) -> ir.ValueRef:
        if isinstance(expr, ir.ADD):
            # prefer negated version
            for left, right in itertools.permutations(expr.subexprs):
                if isinstance(left, ir.USUB) and isinstance(left.operand, ir.MULT):
                    if self.is_float(left):
                        a, b = left.operand.subexprs
                        return ir.MultiplyNegateAdd(a, b, right)
            for left, right in itertools.permutations(expr.subexprs):
                if isinstance(left, ir.MULT):
                    if self.is_float(left):
                        a, b = left.subexprs
                        return ir.MultiplyAdd(a, b, right)
        elif isinstance(expr, ir.SUB):
            # again prefer negated version
            for left, right in itertools.permutations(expr.subexprs):
                if isinstance(left, ir.USUB) and isinstance(left.operand, ir.MULT):
                    if self.is_float(left):
                        a, b = left.operand.subexprs
                        return ir.MultiplyNegateSub(a, b, right)
            for left, right in itertools.permutations(expr.subexprs):
                if isinstance(left, ir.MULT):
                    if self.is_float(left):
                        a, b = left.subexprs
                        return ir.MultiplySub(a, b, right)
        return expr


const_ops = {
    ir.ADD: operator.add,
    ir.BITAND: operator.and_,
    ir.BITOR: operator.or_,
    ir.BITXOR: operator.xor,
    ir.EQ: operator.eq,
    ir.FLOORDIV: operator.floordiv,
    ir.GE: operator.ge,
    ir.GT: operator.gt,
    ir.LE: operator.le,
    ir.LSHIFT: operator.lshift,
    ir.LT: operator.lt,
    ir.MATMULT: operator.matmul,
    ir.MOD: operator.mod,
    ir.MULT: operator.mul,
    ir.NE: operator.ne,
    ir.POW: operator.pow,
    ir.RSHIFT: operator.rshift,
    ir.SUB: operator.sub,
    ir.TRUEDIV: operator.truediv,
    ir.UINVERT: operator.invert,
    ir.USUB: operator.neg,
    ir.SQRT: math.sqrt
}


@singledispatch
def simplify_arithmetic(node):
    msg = f'No method available for arithmetic simplification: "{node}".'
    raise TypeError(msg)


@simplify_arithmetic.register
def _(node: ir.ValueRef):
    return node


@simplify_arithmetic.register
def _(node: ir.Expression):
    return node.reconstruct(*(simplify_arithmetic(subexpr) for subexpr in node.subexprs))


@simplify_arithmetic.register
def _(node: ir.POW):
    if isinstance(node.right, ir.CONSTANT):
        if node.right == ir.Two:
            if isinstance(node.right.value, np.integer):
                return ir.MULT(node.left, node.left)
        elif node.right == ir.Zero:
            return ir.One
        elif node.right == ir.NegativeOne:
            return ir.TRUEDIV(ir.One, node.left)
        elif node.right == ir.NegativeTwo:
            return ir.TRUEDIV(ir.One, ir.MULT(node.left, node.left))
        elif node.right == ir.Half:
            return ir.SQRT(node.left)
    return node


@simplify_arithmetic.register
def _(node: ir.SUB):
    if isinstance(node.right, ir.USUB):
        node = ir.ADD(node.left, node.right)
    return node


@simplify_arithmetic.register
def _(node: ir.ADD):
    for left, right in itertools.permutations(node.subexprs):
        if isinstance(right, ir.USUB):
            if not isinstance(left, ir.USUB):
                # ensure unambiguous
                node = ir.SUB(left, right)
            break
    return node


@simplify_arithmetic.register
def _(node: ir.USUB):
    operand = simplify_arithmetic(node.operand)
    if isinstance(operand, ir.USUB):
        return operand.operand
    return ir.USUB(operand)


@singledispatch
def simplify_integer_reduction(node, typer: TypeHelper):
    return node


@simplify_integer_reduction.register
def _(node: ir.MinReduction, typer: TypeHelper):
    # numpy won't coerce sets correctly
    constants = []
    varying = []
    # check
    for v in node.values:
        if isinstance(v, ir.CONSTANT):
            constants.append(v.value)
        else:
            varying.append(v)
    if len(constants) > 0:
        varying.append(ir.wrap_constant(np.min(varying)))
    return node.reconstruct(*varying)


@simplify_integer_reduction.register
def _(node: ir.MaxReduction, typer: TypeHelper):
    # Check for non-integer
    for subexpr in node.subexprs:
        t = typer(subexpr)
        if isinstance(t, ir.ArrayType):
            if is_integer(t.dtype):
                msg = f"Array reference {subexpr} found in scalar reduction."
                raise CompilerError(msg)
        else:
            if is_integer(t):
                return node

    # numpy won't coerce sets correctly
    constants = []
    varying = []
    for v in node.values:
        if isinstance(v, ir.CONSTANT):
            constants.append(v.value)
        else:
            varying.append(v)
    if len(constants) > 0:
        varying.append(ir.wrap_constant(np.max(varying)))
    return node.reconstruct(*varying)
