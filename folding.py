import itertools
import math
import operator

from functools import singledispatch
from typing import Iterable, Iterator, Union

import numpy as np

import ir

from errors import CompilerError
from type_checks import is_integer, TypeHelper


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


def flatten_nested_index(node: ir.Subscript, typer: TypeHelper):
    """
    Intended to turn a[i][j] into a[i,j]
    Todo: Getting multi-dimensional to actually behave properly may require significant work.
    :param node:
    :param typer:
    :return:
    """
    if not isinstance(node, ir.Subscript):
        return node
    if (isinstance(node.value, ir.Subscript)
            and not isinstance(node.index, ir.Slice)
            and not isinstance(node.value.index, ir.Slice)):
        terms = [node.index, node.value.index]
        next_term = node.value.value
        while isinstance(next_term, ir.Subscript):
            if isinstance(next_term, ir.Slice):
                break
            terms.append(next_term.index)
            next_term = next_term.value
        index = concatenate(reversed(terms))
        # check that this isn't an unsupported indirect indexing
        uses_mask_indices = False
        for e in index.subexprs:
            t = typer.check_type(e)
            if isinstance(t, ir.ArrayType):
                if is_integer(t.dtype):
                    msg = f"Indirect indexing is unsupported {node}."
                    raise CompilerError(msg)
                uses_mask_indices = True
        if uses_mask_indices is False:
            return ir.Subscript(next_term, index)
    return node


def simplify_pow(node: ir.POW):
    """
    slightly more aggressive pow optimizers
    :param node:
    :return:
    """
    if node.right == ir.Two:
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


@singledispatch
def fold_identity(node):
    return node


@fold_identity.register
def _(node: ir.ADD):
    return node.left if node.right == ir.Zero else node.right if node.left == ir.Zero else node


@fold_identity.register
def _(node: ir.FLOORDIV):
    if node.right == ir.One:
        return node.left
    return node


@fold_identity.register
def _(node: ir.MULT):
    return node.left if (node.right == ir.One) else node.right if (node.left == ir.One) else node


@fold_identity.register
def _(node: ir.POW):
    return node.left if node.right == ir.One else node


@fold_identity.register
def _(node: ir.SUB):
    return node.left if node.right == ir.Zero else node


@fold_identity.register
def _(node: ir.TRUEDIV):
    return node.left if node.right == ir.One else node


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
        t = self.typer.check_type(expr)
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


def fold_unary_logic(node: ir.ValueRef):
    """
    Sanitize a couple cases of unary logic.
    :param node:
    :return:
    """
    if isinstance(node, ir.Expression):
        if isinstance(node, ir.ADD):
            for left, right in itertools.permutations(node.subexprs):
                # too ambiguous if both are unary
                if isinstance(right, ir.USUB):
                    return ir.SUB(left, right.operand)
        elif isinstance(node, ir.SUB) and isinstance(node.right, ir.USUB):
            return ir.ADD(node.left, node.right)
        elif isinstance(node, ir.USUB) and isinstance(node.operand, ir.USUB):
            return node.operand.operand
        elif isinstance(node, ir.UINVERT) and isinstance(node.operand, ir.UINVERT):
            return node.operand.operand
    return node


def test_reduction_is_integer(node: Union[ir.MinReduction, ir.MaxReduction], typer: TypeHelper):
    if not isinstance(node, (ir.MinReduction, ir.MaxReduction)):
        msg = f"Cannot treat node {node} as min/max reduction."
        raise TypeError(msg)
    for subexpr in node.subexprs:
        t = typer.check_type(subexpr)
        if isinstance(t, ir.ArrayType):
            t = t.dtype
        if not is_integer(t):
            return False
    return True


@singledispatch
def simplify_integer_reduction(node, typer: TypeHelper):
    return node


@simplify_integer_reduction.register
def _(node: ir.MinReduction, typer: TypeHelper):
    # numpy won't coerce sets correctly
    constants = []
    varying = []
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
        t = typer.check_type(subexpr)
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


@singledispatch
def fold_op(node):
    return node


@fold_op.register
def _(node: ir.MIN):
    for term in node.subexprs:
        if isinstance(term, ir.CONSTANT) and term.is_nan:
            return node.right
    if isinstance(node.left, ir.CONSTANT) and isinstance(node.right, ir.CONSTANT):
        if node.left.is_bool != node.right.is_bool:
            msg = f"Taking min of boolean and non boolean is unsupported for {node.left} and {node.right}."
            raise CompilerError(msg)
        return ir.wrap_constant(min(node.left.value, node.right.value))
    elif node.left == node.right:
        return node.left


@fold_op.register
def _(node: ir.MAX):
    for term in node.subexprs:
        if isinstance(term, ir.CONSTANT) and term.is_nan:
            return node.right
    if isinstance(node.left, ir.CONSTANT) and isinstance(node.right, ir.CONSTANT):
        if node.left.is_bool != node.right.is_bool:
            msg = f"Taking min of boolean and non boolean is unsupported for {node.left} and {node.right}."
            raise CompilerError(msg)
        return ir.wrap_constant(max(node.left.value, node.right.value))
    elif node.left == node.right:
        return node.left


@fold_op.register
def _(node: ir.ADD):
    const_folded = fold_constants(node)
    identity_folded = fold_identity(const_folded)
    return identity_folded


@fold_op.register
def _(node: ir.SUB):
    const_folded = fold_constants(node)
    identity_folded = fold_identity(const_folded)
    return identity_folded


@fold_op.register
def _(node: ir.MULT):
    const_folded = fold_constants(node)
    identity_folded = fold_identity(const_folded)
    return identity_folded


@fold_op.register
def _(node: ir.FLOORDIV):
    if node.right == ir.One:
        return node.left
    return fold_constants(node)


@fold_op.register
def _(node: ir.MOD):
    # safe and common case
    if node.right == ir.One:
        return ir.Zero
    return fold_constants(node)


@fold_op.register
def _(node: ir.TRUTH):
    operand = node.operand
    if isinstance(operand, ir.CONSTANT):
        if operand.is_bool:
            return operand
        else:
            value = np.bool_(operand.value)
            return ir.wrap_constant(value)
    elif isinstance(operand, ir.BoolOp):
        return operand
    return node


@fold_op.register
def _(node: ir.SELECT):
    if isinstance(node.predicate, ir.CONSTANT):
        return node.on_true if node.predicate else node.on_false
    elif isinstance(node.on_true, ir.CONSTANT) and isinstance(node.on_false, ir.CONSTANT):
        if node.on_true == node.on_false or (node.on_true.is_nan and node.on_false.is_nan):
            return node.on_true
    return node


def simplify_arithmetic(node: ir.ValueRef, typer: TypeHelper):
    t = typer.check_type(node)
    simplified = fold_op(node)
    if simplified == node:
        # avoid leaking copies if unchanged
        return node
    simplified_t = typer.check_type(simplified)
    if simplified_t != t:
        simplified = ir.CAST(simplified, t)
    return simplified


def simplify_select(node: ir.SELECT):
    """
    Todo: integrate with other simplifies so we can lose the _class mangling
    :param node:
    :return:
    """
    if isinstance(node.predicate, ir.CONSTANT):
        if operator.truth(node.predicate):
            return node.on_true
        else:
            return node.on_false
    predicate = node.predicate
    on_true = node.on_true
    on_false = node.on_false
    if isinstance(node.predicate, (ir.LT, ir.LE)):
        left, right = node.predicate.subexprs
        if on_true == left and on_false == right:
            return ir.MIN(on_true, on_false)
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
            return ir.MAX(on_false, on_true)
    elif isinstance(predicate, (ir.GT, ir.GE)):
        if on_true == predicate.left and on_false == predicate.right:
            return ir.MAX(on_true, on_false)
        elif on_true == predicate.right and on_false == predicate.left:
            # right if left < right else left
            return ir.MIN(on_false, on_true)
    return node
