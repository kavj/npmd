from copy import copy
from functools import singledispatch
from typing import Iterable

import npmd.ir as ir

from npmd.errors import CompilerError
from npmd.pretty_printing import PrettyFormatter


def map_ids_to_statements(*stmt_seqs: Iterable[ir.StmtBase]):
    id_to_stmt = {}
    groups = []
    for group in stmt_seqs:
        in_group = set()
        for stmt in group:
            stmt_id = id(stmt)
            in_group.add(stmt_id)
            if stmt_id not in id_to_stmt:
                id_to_stmt[stmt_id] = copy(stmt)
        groups.append(in_group)
    return id_to_stmt, groups


def statement_intersection(a: Iterable[ir.StmtBase], b: Iterable[ir.StmtBase]):
    id_to_stmt, groups = map_ids_to_statements(a, b)
    in_a, in_b = groups
    inter = [id_to_stmt[id_] for id_ in in_a.intersection(in_b)]
    return inter


def statement_difference(a: Iterable[ir.StmtBase], b: Iterable[ir.StmtBase]):
    id_to_stmt, groups = map_ids_to_statements(a, b)
    in_a, in_b = groups
    in_first_only = []
    for stmt_id in in_a:
        if stmt_id not in in_b:
            in_first_only.append(id_to_stmt[stmt_id])
    return in_first_only


def is_entry_point(stmt: ir.StmtBase):
    return isinstance(stmt, (ir.IfElse, ir.ForLoop, ir.WhileLoop))


def is_basic_assign(stmt: ir.StmtBase) -> bool:
    """
    Returns True if this is both an assignment and a simple binding operation.

    :param stmt:
    :return:
    """
    return isinstance(stmt, ir.Assign) and isinstance(stmt.target, ir.NameRef)


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
    return node.name


@extract_name.register
def _(node: ir.Call):
    return node.func


def unpack_iterated(target, iterable):
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
            formatter = PrettyFormatter()
            fiterable = formatter(iterable)
            ftarget = formatter(target)
            msg = f"Zip constructs must be fully unpacked. \"{fiterable}\" cannot be unpacked to \"{ftarget}\"."
            raise CompilerError(msg)
    elif isinstance(iterable, ir.Enumerate):
        if isinstance(target, ir.TUPLE):
            first_target, sec_target = target.elements
            yield first_target, ir.AffineSeq(iterable.start, None, ir.One)
            yield from unpack_iterated(sec_target, iterable.iterable)
        else:
            formatter = PrettyFormatter()
            ftarget = formatter(target)
            msg = f"Only tuples are supported unpacking targets. Received \"{ftarget}\" with type {type(target)}."
            raise CompilerError(msg)
    else:
        # Array or sequence reference, with a single opaque target.
        yield target, iterable
