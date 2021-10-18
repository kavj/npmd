import numpy as np

from collections.abc import Iterable

import type_resolution as tr

from driver import ArrayArg
from errors import CompilerError
from utils import wrap_input


def type_from_numpy_type(t: type):
    if t == np.int32:
        return tr.Int32
    elif t == np.int64:
        return tr.Int64
    elif t == np.bool_:
        return tr.BoolType
    elif t == bool:
        return tr.BoolType
    elif t == np.float32:
        return tr.Float32
    elif t == np.float64:
        return tr.Float64
    elif t in (tr.Int32, tr.Int64, tr.Float32, tr.Float64):
        return t
    else:
        msg = f"{t} is not a currently supported type."
        raise CompilerError(msg)


def type_from_array_spec(dims, dtype):
    """
    dims: a tuple of integer constants and strings. Any string must be a valid variable name
    dtype: a numpy scalar type

    """
    if isinstance(dims, Iterable):
        dims = tuple(wrap_input(d) for d in dims)
    else:
        dims = wrap_input(dims),
    dtype = type_from_numpy_type(dtype)
    return ArrayArg(dims, dtype)


def scalar_type_from_spec(bits, is_integral, is_boolean):
    if all(bits != b for b in (8, 32, 64)):
        msg = f"Only 8, 32, and 64 bit types are presently supported."
        raise CompilerError(msg)

