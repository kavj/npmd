from copy import copy
from functools import singledispatch
from typing import Iterable, Union

import lib.ir as ir

from lib.errors import CompilerError
from lib.pretty_printing import PrettyFormatter


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


def unpack_entries(target: ir.TUPLE, iterable: ir.ValueRef):
    if isinstance(target, ir.TUPLE):
        if isinstance(iterable, ir.Zip):
            if len(target.elements) == len(iterable.elements):
                return zip(target.elements, iterable.elements)
        elif isinstance(iterable, ir.Enumerate):
            if len(target.elements) == 2:
                return iter(((target.elements[0], ir.AffineSeq(iterable.start, None, ir.One)), (target.elements[1], iterable.iterable)))
    msg = f'Unable to unpack expression pair {target}, {iterable}.'
    raise CompilerError(msg)


def unpack_iterated(node: ir.ForLoop):
    iterable = node.iterable
    target = node.target
    if isinstance(target, ir.TUPLE):
        queued = [unpack_entries(target, iterable)]
        while queued:
            for target, iterable in queued.pop():
                if isinstance(target, ir.TUPLE):
                    queued.append(unpack_entries(target, iterable))
                    break
                elif isinstance(iterable, (ir.Enumerate, ir.Zip)):
                    msg = f'Unable to unpack. Received target {target} and iterable {iterable}.'
                    raise CompilerError(msg)
                else:
                    yield target, iterable
    else:
        yield target, iterable
