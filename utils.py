import builtins
import itertools
import keyword
import numbers

from functools import singledispatch

import ir
from errors import CompilerError
from visitor import walk

reserved_names = frozenset(set(dir(builtins)).union(set(keyword.kwlist)))


signed_integer_range = {p: (-(2**(p-1)-1), 2**(p-1)-1) for p in (8, 32, 64)}


def get_expr_parameters(expr):
    return {subexpr for subexpr in walk(expr) if isinstance(subexpr, ir.NameRef)}


def is_valid_identifier(name):
    if isinstance(name, ir.NameRef):
        name = name.name
    return isinstance(name, str) and name.isidentifier() and (name not in reserved_names)


@singledispatch
def extract_name(node):
    msg = f"Cannot extract name from node of type {type(node)}. This is probably a bug."
    raise TypeError(msg)


@extract_name.register
def _(node: ir.NameRef):
    return node.name


@extract_name.register
def _(node: ir.Function):
    name = node.name
    if isinstance(name, ir.NameRef):
        name = name.name
    return name


@extract_name.register
def _(node: ir.Call):
    name = node.func
    if isinstance(name, ir.NameRef):
        name = name.name
    return name


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


@singledispatch
def wrap_input(value):
    msg = f"No method to wrap {value} of type {type(value)}."
    raise NotImplementedError(msg)


@wrap_input.register
def _(value: str):
    if not is_valid_identifier(value):
        msg = f"{value} is not a valid variable name."
        raise ValueError(msg)
    return ir.NameRef(value)


@wrap_input.register
def _(value: ir.NameRef):
    return value


@wrap_input.register
def _(value: ir.Constant):
    return value


@wrap_input.register
def _(value: int):
    return ir.IntConst(value)


@wrap_input.register
def _(value: bool):
    return ir.BoolConst(value)


@wrap_input.register
def _(value: numbers.Integral):
    if value == 0:
        v = ir.Zero
    elif value == 1:
        v = ir.One
    else:
        v = ir.IntConst(value)
    return v


@wrap_input.register
def _(value: numbers.Real):
    return ir.FloatConst(value)


@wrap_input.register
def _(value: tuple):
    elts = tuple(wrap_input(elt) for elt in value)
    return ir.Tuple(elts)


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


def unpack_iterated(target, iterable, include_enumerate_indices=True):
    if isinstance(iterable, ir.Zip):
        # must unpack
        if isinstance(target, ir.Tuple):
            if len(target.elements) == len(iterable.elements):
                for t, v in zip(target.elements, iterable.elements):
                    yield from unpack_iterated(t, v)
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
                if include_enumerate_indices:
                    # enumerate is special, because it doesn't add
                    # constraints
                    yield sec_target, ir.AffineSeq(iterable.start, None, ir.One)
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

    if isinstance(expr, ir.BinOp) and expr.op in itertools.chain(ir.add_ops, ir.subtract_ops):
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
    return isinstance(node, ir.BinOp) and node.op in ir.add_ops


def is_subtraction(node):
    return isinstance(node, ir.BinOp) and node.op in ir.subtract_ops


def is_multiplication(node):
    return isinstance(node, ir.BinOp) and node.op in ir.multiply_ops


def is_bit_shift(node):
    return isinstance(node, ir.BinOp) and node.op in ir.bit_shift_ops


def is_logical_op(node):
    return isinstance(node, ir.BoolOp)


def is_compare(node):
    """
    This tests whether we have either a single compare or a chained comparison.
    Chained comparisons are regarded as logical and nodes, where every operand
    is a compare operation and for any consecutive comparisons, indexed by 'i' and 'i+1':
        operands[i].right == operands[i+1].left

    Note, this will ignore cases that are not laid out like:

        (a cmp b) and (b cmp c) and (c cmp d)

    but the internal passes attempt to factor everything this way.

    """

    if isinstance(node, ir.CompareOp):
        return True

    elif isinstance(node, ir.AND):
        first = node.operands[0]
        if not isinstance(first, ir.CompareOp):
            return False
        prev_rhs = first.right
        for operand in itertools.islice(node.operands, 1, None):
            if not isinstance(operand, ir.CompareOp) or prev_rhs != operand.left:
                return False

    else:
        return False

    return True


def is_division(node):
    return isinstance(node, ir.BinOp) and node.op in ir.divide_ops


def is_floor_divide(node):
    return isinstance(node, ir.BinOp) and node.op in ir.floor_divide_ops


def is_true_divide(node):
    return isinstance(node, ir.BinOp) and node.op in ir.true_divide_ops


def is_truth_test(expr):
    return isinstance(expr, (ir.TRUTH, ir.AND, ir.OR, ir.NOT, ir.BoolConst))


def equals_unary_negate(node):
    return isinstance(node, ir.UnaryOp) and node.op == "-"
