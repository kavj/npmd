import builtins
import keyword

import numpy as np

from functools import singledispatch
from typing import Any, List, Tuple

import ir

from errors import CompilerError

reserved_names = frozenset(set(dir(builtins)).union(set(keyword.kwlist)))


def is_assignment(node: ir.StmtBase):
    return isinstance(node, (ir.Assign, ir.InPlaceOp))


def is_entry_point(stmt: ir.StmtBase):
    return isinstance(stmt, (ir.IfElse, ir.ForLoop, ir.WhileLoop))


def has_nan(node: ir.Expression):
    for subexpr in node.subexprs:
        if isinstance(subexpr, ir.CONSTANT):
            if np.isnan(subexpr.value):
                return True


def contains_stmt_types(stmts: List[ir.StmtBase], stmt_types: Tuple[Any, ...]):
    for stmt in stmts:
        if any(isinstance(stmt, stmt_type) for stmt_type in stmt_types):
            return True
        elif isinstance(stmt, (ir.ForLoop, ir.WhileLoop)):
            if type(stmt) in stmt_types:
                return True
            elif contains_stmt_types(stmt.body, stmt_types):
                return True
        elif isinstance(stmt, ir.IfElse):
            if ir.IfElse in stmt_types:
                return True
            elif contains_stmt_types(stmt.if_branch, stmt_types) or contains_stmt_types(stmt.else_branch, stmt_types):
                return True
    return False


def get_stmt_types(stmts: List[ir.StmtBase], stmt_types: Tuple[Any, ...]):
    retrievals = []
    for stmt in stmts:
        if any(isinstance(stmt, stmt_type) for stmt_type in stmt_types):
            retrievals.append(stmt)
        if isinstance(stmt, (ir.ForLoop, ir.WhileLoop)):
            retrievals.extend(get_stmt_types(stmt.body, stmt_types))
        elif isinstance(stmt, ir.IfElse):
            retrievals.extend(get_stmt_types(stmt.if_branch, stmt_types))
            retrievals.extend(get_stmt_types(stmt.else_branch, stmt_types))
    return retrievals


def is_allowed_identifier(name):
    if isinstance(name, ir.NameRef):
        name = name.name
    return isinstance(name, str) and name.isidentifier() and (name not in reserved_names)


@singledispatch
def extract_name(node):
    msg = f"Cannot extract name from node of type {type(node)}. This is probably a bug."
    raise TypeError(msg)


@extract_name.register
def _(node: str):
    return node


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


def unpack_assignment(target, value, pos):
    if isinstance(target, ir.TUPLE) and isinstance(value, ir.TUPLE):
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
        if isinstance(target, ir.TUPLE):
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
        if isinstance(target, ir.TUPLE):
            first_target, sec_target = target.elements
            if include_enumerate_indices:
                # enumerate is special, because it doesn't add
                # constraints
                yield first_target, ir.AffineSeq(iterable.start, None, ir.One)
            yield from unpack_iterated(sec_target, iterable.iterable)
        else:
            msg = f"Only tuples are supported unpacking targets. Received {type(target)}."
            raise CompilerError(msg)
    else:
        # Array or sequence reference, with a single opaque target.
        yield target, iterable


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
        operand_iter = iter(node.operands)
        first = next(operand_iter)
        if not isinstance(first, ir.CompareOp):
            return False
        prev_rhs = first.right
        for operand in operand_iter:
            if not isinstance(operand, ir.CompareOp) or prev_rhs != operand.left:
                return False

    else:
        return False

    return True
