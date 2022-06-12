import itertools
import math
import operator

from functools import singledispatch, singledispatchmethod
from typing import Callable, Iterable, Iterator, Union

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


class OpFold:

    int_dtypes = (ir.float32, ir.float64)

    def check_type(self, expr: ir.ValueRef):
        return self.typer.check_type(expr)

    def check_dtype(self, expr: ir.ValueRef):
        return self.typer.check_dtype(expr)

    def prepare_replacement(self, expr: ir.ValueRef, base: ir.ValueRef):
        if expr == base:
            return base
        base_type = self.typer.check_type(base)
        expr_type = self.typer.check_type(expr)
        if base_type != expr_type:
            expr = ir.CAST(expr, base_type)
        return expr

    def is_constant(self, node):
        if isinstance(node, ir.CONSTANT):
            return True
        elif isinstance(node, ir.Expression):
            return all(isinstance(subexpr, ir.Expression) for subexpr in node.subexprs)
        return False

    def __init__(self, typer: TypeHelper):
        self.typer = typer

    def __call__(self, node):
        return self.visit(node)

    @singledispatchmethod
    def visit(self, node):
        msg = f'No method to fold identity for "{node}".'
        raise TypeError(msg)

    @visit.register
    def _(self, node: ir.ValueRef):
        return node

    @visit.register
    def _(self, node: ir.ADD):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if right == ir.Zero:
            repl = left
        else:
            repl = fold_constants(node.reconstruct(left, right))
        if isinstance(repl, ir.ADD):
            left = repl.left
            right = repl.right
            if isinstance(right, ir.USUB):
                repl = ir.SUB(left, right.operand)
            elif isinstance(left, ir.USUB):
                repl = ir.SUB(right, left.operand)
        return self.prepare_replacement(repl, node)

    @visit.register
    def _(self, node: ir.MULT):
        left = self.visit(node.left)
        right = self.visit(node.right)
        for term in (left, right):
            if ir.is_nan(term):
                return term
        if right == ir.Zero:
            repl = ir.Zero
        elif right == ir.One:
            repl = left
        else:
            repl = node.reconstruct(left, right)
        return self.prepare_replacement(repl, node)

    @visit.register
    def _(self, node: ir.SUB):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if right == ir.Zero:
            repl = left
        elif left == ir.Zero:
            if isinstance(right, ir.USUB):
                # avoid double unary negative
                repl = right.operand
            else:
                if isinstance(right, ir.USUB):
                    repl = right
                else:
                    repl = ir.USUB(right)
        else:
            repl = fold_constants(node.reconstruct(left, right))
        if isinstance(repl, ir.SUB):
            if isinstance(repl.right, ir.USUB):
                repl = ir.ADD(repl.left, repl.right.operand)
        return self.prepare_replacement(repl, node)

    @visit.register
    def _(self, node: ir.CAST):
        # find redundant ones
        unwrapped = node
        while isinstance(unwrapped, ir.CONSTANT):
            unwrapped = unwrapped.value
        if unwrapped != node.value:
            return node.reconstruct(unwrapped)
        return node

    @visit.register
    def _(self, node: ir.FLOORDIV):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if right == ir.One:
            repl = left
        else:
            repl = fold_constants(node.reconstruct(left, right))
        base_dtype = self.check_dtype(node)
        if base_dtype not in self.int_dtypes:
            # Todo: need floor for this
            return node
        return self.prepare_replacement(repl, node)

    @visit.register
    def _(self, node: ir.AND):
        operands = []
        for subexpr in node.subexprs:
            subexpr = self.visit(subexpr)
            if isinstance(subexpr, ir.CONSTANT):
                # return False or ignore
                if not operator.truth(subexpr):
                    return ir.FALSE
            else:
                operands.append(subexpr)
        if len(operands) == 0:
            return ir.TRUE
        elif len(operands) == 1:
            operand = operands.pop()
            if not isinstance(operand, ir.TRUTH):
                operand = ir.TRUTH(operand)
            return operand
        else:
            return node.reconstruct(*operands)

    @visit.register
    def _(self, node: ir.OR):
        operands = []
        for subexpr in node.subexprs:
            subexpr = self.visit(subexpr)
            if isinstance(subexpr, ir.CONSTANT):
                if operator.truth(subexpr):
                    return ir.TRUE
            else:
                operands.append(subexpr)
        if len(operands) == 0:
            return ir.FALSE
        elif len(operands) == 1:
            operand = operands.pop()
            if not isinstance(operand, ir.TRUTH):
                operand = ir.TRUTH(operand)
            return operand
        else:
            return node.reconstruct(*operands)

    @visit.register
    def _(self, node: ir.UINVERT):
        pass

    @visit.register
    def _(self, node: ir.USUB):
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

    @visit.register
    def _(self, node: ir.MOD):
        left = self.visit(node.left)
        right = self.visit(node.right)
        repl = fold_constants(node.reconstruct(left, right))
        return self.prepare_replacement(repl, node)

    @visit.register
    def _(self, node: ir.TRUEDIV):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if node.right == ir.One:
            repl = left
        else:
            repl = node.reconstruct(left, right)
        return self.prepare_replacement(repl, node)

    @visit.register
    def _(self, node: ir.POW):
        # have to handle casts for cases like some_int ** 1.0
        repl = simplify_pow(node)
        return self.prepare_replacement(repl, node)

    @visit.register
    def _(self, node: ir.SELECT):
        on_true = self.visit(node.on_true)
        on_false = self.visit(node.on_false)
        predicate = self.visit(node.predicate)
        if isinstance(predicate, ir.CONSTANT):
            repl = on_true if operator.truth(predicate) else on_false
        else:
            repl = node
            if isinstance(predicate, (ir.LT, ir.LE)):
                left = predicate.left
                right = predicate.right
                if on_true == left and on_false == right:
                    repl = ir.MIN(on_true, on_false)
                elif on_false == right and on_true == left:
                    repl = ir.MAX(on_false, on_true)
            elif isinstance(predicate, (ir.GT, ir.GE)):
                left = predicate.left
                right = predicate.right
                if on_true == left and on_false == right:
                    repl = ir.MAX(on_true, on_false)
                elif on_true == right and on_false == left:
                    repl = ir.MIN(on_false, on_true)
        return self.prepare_replacement(repl, node)

    @visit.register
    def _(self, node: ir.CompareOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        repl = node.reconstruct(left, right)
        return self.prepare_replacement(repl, node)


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
