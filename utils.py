import builtins
import itertools
import keyword

from functools import singledispatch
from typing import Any, List, Tuple

import ir

from errors import CompilerError
from visitor import walk

reserved_names = frozenset(set(dir(builtins)).union(set(keyword.kwlist)))


def contains_stmt_types(stmts: List[ir.StmtBase], stmt_types: Tuple[Any, ...]):
    for stmt in stmts:
        if any(isinstance(stmt, stmt_type) for stmt_type in stmt_types):
            return True
        elif isinstance(stmt, (ir.ForLoop, ir.WhileLoop)):
            if contains_stmt_types(stmt.body, stmt_types):
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


def contains_loops(stmts: List[ir.StmtBase]):
    return contains_stmt_types(stmts, (ir.ForLoop, ir.WhileLoop))


def contains_break_or_return(stmts: List[ir.StmtBase]):
    return contains_stmt_types(stmts, (ir.Break, ir.Return))


def diverges(stmts: List[ir.StmtBase]):
    return contains_stmt_types(stmts, (ir.ForLoop, ir.WhileLoop, ir.IfElse, ir.Break, ir.Return, ir.Continue))


def contains_branches(stmts: List[ir.StmtBase]):
    return contains_stmt_types(stmts, (ir.IfElse,))


def get_expr_parameters(expr):
    return {subexpr for subexpr in walk(expr) if isinstance(subexpr, ir.NameRef)}


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
def _(node: ir.ArrayRef):
    return node.name.name


@extract_name.register
def _(node: ir.ScalarRef):
    return node.name.name


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


def is_fma_pattern(expr):
    """
    This ignores safety issues, which may be addressed later for anything
    that looks like a * b - c * d.


    """

    if isinstance(expr, (ir.ADD, ir.SUB)):
        left = expr.left
        right = expr.right
        for operand in (left, right):
            if isinstance(operand, ir.BinOp):
                if isinstance(operand, ir.MULT):
                    return True
            elif isinstance(operand, ir.UnaryOp):
                # Expression simplifiers should have already folded any multiple
                # nestings of unary -
                if isinstance(operand, ir.USUB) and isinstance(operand.operand, ir.MULT):
                    return True
    return False


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


def is_truth_test(expr):
    return isinstance(expr, (ir.TRUTH, ir.AND, ir.OR, ir.NOT, ir.Constant))
