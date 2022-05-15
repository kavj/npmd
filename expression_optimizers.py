import numbers
import operator

import numpy as np

from dataclasses import dataclass
from functools import cache, singledispatch
from itertools import permutations

import ir

from errors import CompilerError
from symbol_table import symbol_table
from type_checks import TypeHelper


def has_nan(node: ir.Expression):
    for subexpr in node.subexprs:
        if isinstance(subexpr, ir.Constant):
            if np.isnan(subexpr.value):
                return True


@cache
def is_constant(node: ir.ValueRef):
    if isinstance(node, ir.Constant):
        return True
    if isinstance(node, ir.AND):
        for subexpr in node.subexprs:
            if isinstance(subexpr, ir.Constant) and not operator.truth(subexpr):
                return True
    elif isinstance(node, ir.OR):
        for subexpr in node.subexprs:
            if isinstance(subexpr, ir.Constant) and operator.truth(subexpr):
                return True
    elif isinstance(node, ir.Expression):
        for subexpr in node.subexprs:
            if not is_constant(subexpr):
                return False
        return True
    return False


def can_negate_constant(node: ir.Constant):
    value = node.value
    if value.dtype == np.dtype('bool'):
        return False
    if isinstance(value, float):
        try:
            value = -value  # used to trigger overflow error
            return True
        except OverflowError:
            return False
    else: # checking based on 64 bit int
        return -2**63 < value


@dataclass(frozen=True)
class MultiplyAdd(ir.Expression):
    """
    a * b + c
    """

    a: ir.ValueRef
    b: ir.ValueRef
    c: ir.ValueRef

    @property
    def subexprs(self):
        yield self.a
        yield self.b
        yield self.c


@dataclass(frozen=True)
class MultiplySub(ir.Expression):
    """
    a * b - c
    """

    a: ir.ValueRef
    b: ir.ValueRef
    c: ir.ValueRef

    @property
    def subexprs(self):
        yield self.a
        yield self.b
        yield self.c


@dataclass(frozen=True)
class MultiplyNegateAdd(ir.Expression):
    """
    -(a * b) + c
    """

    a: ir.ValueRef
    b: ir.ValueRef
    c: ir.ValueRef

    @property
    def subexprs(self):
        yield self.a
        yield self.b
        yield self.c


@dataclass(frozen=True)
class MultiplyNegateSub(ir.Expression):
    """
    a * b - c
    """

    a: ir.ValueRef
    b: ir.ValueRef
    c: ir.ValueRef

    @property
    def subexprs(self):
        yield self.a
        yield self.b
        yield self.c


@singledispatch
def as_multiply_accum(node: ir.Expression):
    return node


@as_multiply_accum.register
def _(node: ir.ADD):
    for left, right in permutations(node.subexprs):
        if isinstance(left, ir.MULT):
            a, b = left.subexprs
            return MultiplyAdd(a, b, right)
        elif isinstance(left, ir.USUB) and isinstance(left.operand, ir.MULT):
            a, b = left.operand.subexprs
            return MultiplyNegateAdd(a, b, right)
    return node


@as_multiply_accum.register
def _(node: ir.SUB):
    for left, right in permutations(node.subexprs):
        if isinstance(left, ir.MULT):
            a, b = left.subexprs
            return MultiplySub(a, b, right)
        elif isinstance(left, ir.USUB) and isinstance(left.operand, ir.MULT):
            a, b = left.operand.subexprs
            return MultiplyNegateSub(a, b, right)
    return node


# Todo: add type promotions


@singledispatch
def fold_constants(node, type_helper: TypeHelper):
    return node


@fold_constants.register
def _(node: ir.USUB, type_helper: TypeHelper):
    if is_constant(node.operand):
        value = fold_constants(node.operand, type_helper)
        return ir.wrap_constant(-value.value)
    return node


@fold_constants.register
def _(node: ir.ADD, type_helper: TypeHelper):
    left, right = node.subexprs
    if isinstance(left, ir.Constant) and isinstance(right, ir.Constant):
        return ir.wrap_constant(left.value + right.value)
    for left, right in permutations((left, right)):
        if right == ir.Zero:
            return left
    return node


@fold_constants.register
def _(node: ir.SUB, type_helper: TypeHelper):
    left, right = node.subexprs
    if isinstance(left, ir.Constant) and isinstance(right, ir.Constant):
        return ir.wrap_constant(left.value - right.value)
    elif left == ir.Zero:
        return ir.USUB(right)
    elif right == ir.Zero:
        return left
    return node


@fold_constants.register
def _(node: ir.MULT, type_helper: TypeHelper):
    left, right = node.subexprs
    if isinstance(left, ir.Constant) and isinstance(right, ir.Constant):
        return ir.wrap_constant(left.value * right.value)
    for left, right in permutations((left, right)):
        if left == ir.Zero:
            return ir.Zero
        elif left == ir.One:
            return right
        elif left == ir.Two:
            return ir.ADD(left, right)
        elif left == ir.NegativeOne:
            return ir.USUB(right)
    return node


@fold_constants.register
def _(node: ir.TRUEDIV):
    # can raise
    left, right = node.subexprs
    if isinstance(left, ir.Constant) and isinstance(right, ir.Constant):
        if right == ir.Zero:
            msg = f"Source contains a zero division expression: {left} / {right}."
            raise CompilerError(msg)
        return ir.wrap_constant(left.value / right.value)
    if left == ir.Zero:
        return ir.Zero
    elif right == ir.One:
        return left
    elif right == ir.NegativeOne:
        return ir.USUB(left)
    elif right == ir.Two:
        return ir.MULT(ir.Half, left)
    elif right == ir.Half:
        return ir.ADD(left, left)
    elif right == ir.NegativeOne:
        return ir.USUB(left)
    return node


@fold_constants.register
def _(node: ir.FLOORDIV):
    left, right = node.subexprs
    if isinstance(left, ir.Constant) and isinstance(right, ir.Constant):
        if right == ir.Zero:
            msg = f"Source contains a zero division expression: {left} // {right}."
            raise CompilerError(msg)
        return ir.wrap_constant(left.value // right.value)
    if left == ir.Zero:
        return ir.Zero
    # the others require flooring for simplification.
    return node


@fold_constants.register
def _(node: ir.AND):
    buffered = []
    for subexpr in node.subexprs:
        if isinstance(subexpr, ir.Constant):
            if not operator.truth(subexpr):
                return ir.FALSE
        else:
            buffered.append(subexpr)
    return ir.AND(*buffered)


@fold_constants.register
def _(node: ir.OR):
    buffered = []
    for subexpr in node.subexprs:
        if isinstance(subexpr, ir.Constant):
            if operator.truth(subexpr):
                return ir.TRUE
        else:
            buffered.append(subexpr)
    return ir.OR(*buffered)


@fold_constants.register
def _(node: ir.MOD):
    if node.right == ir.One:
        return ir.Zero
    elif node.right == ir.Zero:
        msg = f"Expression {node} contains division by zero."
        raise CompilerError(msg)
    elif isinstance(node.left, ir.Constant) and isinstance(node.right, ir.Constant):
        return ir.wrap_constant(np.mod(node.left.value, node.right.value))
    return node


@fold_constants.register
def _(node: ir.POW):
    # Todo: This needs cast nodes, so that something like a**2 comes out as np.float64(a * a)
    if is_constant(node):
        return np.power(node.left, node.right)
    base, exponent = node.subexprs
    if exponent == ir.Zero:
        # numpy has this taking precedence
        if isinstance(exponent.value, numbers.Integral):
            # only fold if no unsafe stuff
            return ir.One
    elif has_nan(node):
        return ir.NAN
    elif base == 0:
        if isinstance(exponent.value, numbers.Integral):
            return ir.Zero
    elif exponent == ir.Half:
        return ir.Sqrt(base)
    elif exponent == ir.One:
        return base
    elif exponent == ir.Two:
        return ir.MULT(base, base)
    return node


@singledispatch
def fold_unary(node, type_helper: TypeHelper):
    return node


@fold_unary.register
def _(node: ir.USUB, type_helper: TypeHelper):
    if isinstance(node.operand, ir.USUB):
        return node.operand.operand


@fold_unary.register
def _(node: ir.ADD, type_helper: TypeHelper):
    # try to convert unary negate into subtraction
    for left, right in permutations(node.subexprs):
        if isinstance(right, ir.USUB) and not isinstance(left, ir.USUB):
            return ir.SUB(left, right)
    return node


@fold_unary.register
def _(node: ir.SUB, type_helper: TypeHelper):
    for left, right in permutations(node.subexprs):
        if isinstance(right, ir.USUB) and not isinstance(left, ir.USUB):
            return ir.ADD(left, right)
    return node


@fold_unary.register
def _(node: ir.MULT, type_helper: TypeHelper):
    for left, right in permutations(node.subexprs):
        if isinstance(right, ir.USUB):
            if isinstance(left, ir.Constant):
                if left.value < 0:
                    if isinstance(left.value, float) or left.value < -(2**63-1):
                        left = ir.wrap_constant(-left.value)
                        return ir.MULT(left, right.operand)


def optimize_select(node: ir.Select):
    if not isinstance(node, ir.Select):
        return node
    predicate, on_true, on_false = node.subexprs
    if isinstance(node.predicate, ir.Constant):
        # partial fold if constant predicate
        if node.predicate.value:
            return node.on_true
        else:
            return node.on_false
    if isinstance(node.predicate, ir.Constant):
        if node.predicate.value:
            return node.on_true
        else:
            return node.on_false
    # predicate is non-const. Check if coercible to Min or max
    if isinstance(predicate, (ir.LT, ir.LE)):
        left, right = predicate.subexprs
        if on_true == left and on_false == right:
            return ir.Min(on_true, on_false)
        elif on_false == right and on_true == left:
            # This is almost negated. The issue is if in the destination assembly:
            #
            #     min(a,b) is implemented as a if a <= b else b
            #     max(a,b) is implemented as a if a >= b else b
            #
            #  which is common, we reverse operand order to properly catch unordered cases
            #  This does not follow Python's min/max conventions, which are too error prone.
            #  Those can arbitrarily propagate or suppress nans as a side effect of
            #  determining type from the leading operand.
            return ir.Max(on_false, on_true)
    elif isinstance(predicate, (ir.GT, ir.GE)):
        if on_true == predicate.left and on_false == predicate.right:
            return ir.Max(on_true, on_false)
        elif on_true == predicate.right and on_false == predicate.left:
            # right if left < right else left
            return ir.Min(on_false, on_true)
    return node


def simplify_expr(node: ir.ValueRef, type_helper: TypeHelper):
    if not isinstance(node, ir.Expression):
        return node
    subexprs = []
    for subexpr in node.subexprs:
        subexpr = simplify_expr(subexpr, type_helper)
        subexprs.append(subexpr)
    subexprs_folded = node.reconstruct(*subexprs)
    const_folded = fold_constants(subexprs_folded, type_helper)
    unary_folded = fold_unary(const_folded, type_helper)
    with_select_opt = optimize_select(unary_folded)
    return with_select_opt
