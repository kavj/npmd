import builtins
import keyword
from dataclasses import dataclass

import ir


@dataclass
class ErrorValue:
    msg: str
    pos: ir.Position


keywords = frozenset(set(keyword.kwlist))
builtin_names = frozenset(set(dir(builtins)))


def shadows_builtin_name(name):
    return name in keywords or name in builtin_names


def check_array_access():
    """
    Check for dimension count mismatches.
    Check that each subscript refers to an array input.

    """
    pass


def find_array_allocation():
    pass


def find_zero_dim_array():
    pass


def find_invalid_step():
    pass


def check_for_loop():
    pass


def find_aliased_array_names():
    pass


def find_bad_call_sigs():
    """

    """
    pass


def find_invalid_slices():
    pass


def is_supported_directive(name, args):
    """
    Check if matches simd or parallel directive

    """
    pass


def find_duplicate_args(func):
    """
    Haven't decided on keyword support...

    """
    args = set()
    dupes = set()
    for arg in func.args:
        if arg in args:
            dupes.add(arg)
        else:
            args.add(arg)
    if dupes:
        return dupes


def find_zero_step_size():
    pass


def find_unsafe_exprs(exprs):
    unsafe = set()
    for e in exprs:
        if isinstance(e, ir.BinOp):
            if e.op in ("/", "//", "/=", "//="):
                if e.right.constant:
                    pass
                    # if e.right.value == 0:
                    #    unsafe.add(e)
    return unsafe

