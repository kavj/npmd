from functools import singledispatch

import npmd.ir as ir

from npmd.errors import CompilerError
from npmd.pretty_printing import PrettyFormatter


def is_entry_point(stmt: ir.StmtBase):
    return isinstance(stmt, (ir.IfElse, ir.ForLoop, ir.WhileLoop))


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
    return node.func.name


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
