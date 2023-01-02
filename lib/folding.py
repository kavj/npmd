import itertools

from functools import singledispatch
from typing import Iterable, Iterator, Union

import numpy as np

import lib.ir as ir

from lib.errors import CompilerError
from lib.traversal import walk
from lib.type_checks import is_integer, TypeHelper


def make_add(a: ir.ValueRef, b: ir.ValueRef):
    if isinstance(a, ir.CONSTANT) and isinstance(b, ir.CONSTANT):
        return ir.wrap_constant(a.value + b.value)
    if isinstance(b, ir.USUB):
        return ir.SUB(a, b)
    elif isinstance(a, ir.USUB):
        return ir.SUB(b, a)
    return ir.ADD(a, b)


def make_sub(a: ir.ValueRef, b: ir.ValueRef):
    if isinstance(a, ir.CONSTANT) and isinstance(b, ir.CONSTANT):
        return ir.wrap_constant(a.value - b.value)
    if isinstance(b, ir.USUB):
        return ir.ADD(a, b)
    return ir.ADD(a, b)


def make_logical_invert(expr: ir.ValueRef):
    """
    Applies logical not with simplification
    :param expr:
    :return:
    """
    if isinstance(expr, ir.CONSTANT):
        return ir.TRUE if expr else ir.FALSE
    if isinstance(expr, ir.NOT):
        return expr.operand
    return ir.NOT(expr)


def make_unary_negate(expr: ir.ValueRef):
    """
    Applies unary minus with simplification
    :param expr:
    :return:
    """
    if isinstance(expr, ir.CONSTANT):
        if expr.is_bool:
            return ir.FALSE if expr == ir.TRUE else ir.TRUE
        else:
            return ir.wrap_constant(-expr.value)
    if isinstance(expr, ir.USUB):
        return expr.operand
    return ir.USUB(expr)


def add_cast(expr: ir.ValueRef, cast_type: Union[ir.ArrayType, np.dtype]):
    repl = expr
    while isinstance(repl, ir.CAST):
        repl = repl.value
    if expr == repl:
        return expr
    return ir.CAST(expr, cast_type)


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


def cast_if_type_differs(node: ir.ValueRef, repl: ir.ValueRef, typer: TypeHelper):
    if isinstance(repl, ir.CAST):
        while isinstance(repl, ir.CAST):
            repl = repl.value
    t_original = typer(node)
    t_repl = typer(repl)
    if t_original != t_repl:
        if isinstance(repl, ir.CONSTANT):
            repl = ir.wrap_constant(repl.value, t_original, repl.is_predicate)
        else:
            repl = ir.CAST(repl, t_original)
    return repl


@singledispatch
def simplify(node, typer):
    msg = f'No method to simplify expression "{node}"'
    raise TypeError(msg)


@simplify.register
def _(node: ir.ValueRef, typer: TypeHelper):
    return node


@simplify.register
def _(node: ir.AffineSeq, typer: TypeHelper):
    repl = []
    for subexpr in node.subexprs:
        subexpr = simplify(subexpr, typer)
        repl.append(subexpr)
    return ir.AffineSeq(*repl)


@simplify.register
def _(node: ir.Expression, typer: TypeHelper):
    repl = node.reconstruct(*(simplify(subexpr, typer) for subexpr in node.subexprs))
    return cast_if_type_differs(node, repl, typer)


@simplify.register
def _(node: ir.CAST, typer: TypeHelper):
    v = node.value
    if isinstance(v, ir.CAST):
        # remove nested casts
        while isinstance(v, ir.CAST):
            v = v.value
    v = simplify(v, typer)
    # only re-add the cast if it needs it
    return cast_if_type_differs(node, v, typer)


@simplify.register
def _(node: ir.ADD, typer: TypeHelper):
    a, b = (simplify(subexpr, typer) for subexpr in node.subexprs)
    if a == ir.NAN or b == ir.NAN:
        return ir.NAN
    if isinstance(a, ir.CONSTANT) and isinstance(b, ir.CONSTANT):
        repl = ir.wrap_constant(a.value + b.value)
    elif a == ir.Zero or b == ir.Zero:
        repl = b if a == ir.Zero else a
    else:
        repl = ir.ADD(a, b)
    return cast_if_type_differs(node, repl, typer)


@simplify.register
def _(node: ir.SUB, typer: TypeHelper):
    a, b = (simplify(subexpr, typer) for subexpr in node.subexprs)
    if a == ir.NAN or b == ir.NAN:
        return ir.NAN
    if isinstance(a, ir.CONSTANT) and isinstance(b, ir.CONSTANT):
        repl = ir.wrap_constant(a.value - b.value)
    elif a == ir.Zero:
        b = ir.USUB(b)
        repl = simplify(b, typer)
    elif b == ir.Zero:
        repl = a
    else:
        repl = ir.SUB(a, b)
    return cast_if_type_differs(node, repl, typer)


@simplify.register
def _(node: ir.MULT, typer: TypeHelper):
    a, b = (simplify(subexpr, typer) for subexpr in node.subexprs)
    if a == ir.NAN or b == ir.NAN:
        return ir.NAN
    if isinstance(a, ir.CONSTANT) and isinstance(b, ir.CONSTANT):
        repl = ir.wrap_constant(a.value * b.value)
    elif a == ir.Zero or b == ir.Zero:
        repl = ir.Zero
    elif a == ir.One or b == ir.One:
        repl = b if a == ir.One else a
    elif a == ir.Two or b == ir.Two:
        other = a if b == ir.Two else b
        repl = ir.ADD(other, other)
    else:
        repl = ir.MULT(a, b)
    return cast_if_type_differs(node, repl, typer)


@simplify.register
def _(node: ir.TRUEDIV, typer: TypeHelper):
    a, b = (simplify(subexpr, typer) for subexpr in node.subexprs)
    if a == ir.NAN or b == ir.NAN:
        return ir.NAN
    if b == ir.Zero:
        raise CompilerError
    if isinstance(a, ir.CONSTANT) and isinstance(b, ir.CONSTANT):
        repl = ir.wrap_constant(a.value / b.value)
    elif a == ir.Zero:
        repl = b
    elif b == ir.One:
        repl = a
    else:
        repl = ir.TRUEDIV(a, b)
    return cast_if_type_differs(node, repl, typer)


@simplify.register
def _(node: ir.FLOORDIV, typer: TypeHelper):
    a, b = (simplify(subexpr, typer) for subexpr in node.subexprs)
    if a == ir.NAN or b == ir.NAN:
        return ir.NAN
    if b == ir.Zero:
        raise CompilerError
    if isinstance(a, ir.CONSTANT) and isinstance(b, ir.CONSTANT):
        repl = ir.wrap_constant(a.value // b.value)
    elif a == ir.Zero:
        repl = b
    elif b == ir.One:
        repl = ir.FLOOR(a)
        repl = simplify(repl, typer)
    else:
        repl = ir.TRUEDIV(a, b)
    return cast_if_type_differs(node, repl, typer)


@simplify.register
def _(node: ir.FLOOR, typer: TypeHelper):
    v = node.value
    while isinstance(v, ir.FLOOR):
        v = node.value
    v = simplify(v, typer)
    repl = ir.FLOOR(v)
    return cast_if_type_differs(node, repl, typer)


@simplify.register
def _(node: ir.USUB, typer: TypeHelper):
    v = node
    while isinstance(v, ir.USUB) and isinstance(v.operand, ir.USUB):
        v = v.operand.operand
    if isinstance(v, ir.USUB):
        operand = simplify(v.operand, typer)
        repl = ir.USUB(operand)
    else:
        repl = simplify(v, typer)
    return cast_if_type_differs(node, repl, typer)


@simplify.register
def _(node: ir.MIN, typer: TypeHelper):
    subexprs = [simplify(subexpr, typer) for subexpr in node.subexprs]
    if all(isinstance(subexpr, ir.CONSTANT) for subexpr in subexprs):
        repl = ir.wrap_constant(max(subexpr.value for subexpr in subexprs))
    else:
        a, b = subexprs
        if a == b:
            repl = a
        else:
            repl = ir.MIN(a, b)
    return cast_if_type_differs(node, repl, typer)


@simplify.register
def _(node: ir.MAX, typer: TypeHelper):
    subexprs = [simplify(subexpr, typer) for subexpr in node.subexprs]
    if all(isinstance(subexpr, ir.CONSTANT) for subexpr in subexprs):
        repl = ir.wrap_constant(max(subexpr.value for subexpr in subexprs))
    else:
        a, b = subexprs
        if a == b:
            repl = a
        else:
            repl = ir.MAX(a, b)
    return cast_if_type_differs(node, repl, typer)


@simplify.register
def _(node: ir.NOT, typer: TypeHelper):
    if isinstance(node.operand, ir.NOT):
        v = node
        while isinstance(v, ir.NOT) and isinstance(v.operand, ir.NOT):
            v = v.operand.operand
        repl = simplify(v, typer)
    else:
        repl = simplify(node.operand, typer)
    repl = ir.NOT(repl)
    return cast_if_type_differs(node, repl, typer)


@simplify.register
def _(node: ir.POW, typer: TypeHelper):
    a, b = (simplify(subexpr, typer) for subexpr in node.subexprs)
    if isinstance(a, ir.CONSTANT) and isinstance(b, ir.CONSTANT):
        repl = a.value ** b.value
        return cast_if_type_differs(node, repl, typer)
    if isinstance(b, ir.CONSTANT):
        if b == ir.Two:
            repl = ir.MULT(a, a)
        elif node.right == ir.Zero:
            repl = ir.One
        elif node.right == ir.NegativeOne:
            repl = ir.TRUEDIV(ir.One, node.left)
        elif node.right == ir.NegativeTwo:
            repl = ir.TRUEDIV(ir.One, ir.MULT(node.left, node.left))
        elif node.right == ir.Half:
            repl = ir.SQRT(a)
        else:
            repl = ir.POW(a, b)
    else:
        repl = ir.POW(a, b)
    return cast_if_type_differs(node, repl, typer)


@simplify.register
def _(node: ir.Subscript, typer: TypeHelper):
    repl = ir.Subscript(*(simplify(subexpr, typer) for subexpr in node.subexprs))
    if isinstance(repl.index, ir.Slice):
        if repl.index == ir.Slice(ir.Zero, ir.SingleDimRef(repl.value, ir.Zero), ir.One):
            repl = repl.value
    return cast_if_type_differs(node, repl, typer)


@simplify.register
def _(node: ir.Slice, typer: TypeHelper):
    subexprs = [simplify(subexpr, typer) for subexpr in node.subexprs]
    repl = ir.Slice(*subexprs)
    return repl

# TODO: may want to consolidate reductions by passing symbol


@simplify.register
def _(node: ir.MaxReduction, typer: TypeHelper):
    subexprs = {simplify(subexpr, typer) for subexpr in node.subexprs}
    if any(isinstance(subexpr, ir.MaxReduction) for subexpr in subexprs):
        repl = set()
        for subexpr in subexprs:
            if isinstance(subexpr, ir.MaxReduction):
                repl.update(subexpr.subexprs)
            else:
                repl.add(subexpr)
        subexprs = repl
    return ir.MaxReduction(subexprs)


@simplify.register
def _(node: ir.MinReduction, typer: TypeHelper):
    subexprs = {simplify(subexpr, typer) for subexpr in node.subexprs}
    if any(isinstance(subexpr, ir.MinReduction) for subexpr in subexprs):
        repl = set()
        for subexpr in subexprs:
            if isinstance(subexpr, ir.MinReduction):
                repl.update(subexpr.subexprs)
            else:
                repl.add(subexpr)
        subexprs = repl
    return ir.MinReduction(*subexprs)


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


def simplify_max_reduction(node: ir.MaxReduction, typer: TypeHelper):
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


def fold_casts(expr: ir.ValueRef, typer: TypeHelper):
    if not isinstance(expr, ir.CAST) and not any(isinstance(subexpr, ir.CAST) for subexpr in walk(expr)):
        return expr
    updated = {}
    for subexpr in walk(expr):
        repl = subexpr
        if any(s in updated for s in repl.subexprs):
            repl = repl.reconstruct(*(updated.get(s, s) for s in repl.subexprs))
        if isinstance(repl, ir.CAST):
            if typer(repl.value) == repl.target_type:
                repl = repl.value
        updated[subexpr] = repl
    return updated[expr]
