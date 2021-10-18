import numpy as np

from collections.abc import Iterable

import ir

import type_resolution as tr

from errors import CompilerError
from utils import wrap_input


def scalar_type_from_numpy_type(t: type):
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


def type_from_array_spec(ndims, dims, dtype):
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
    if is_integral:
        if bits == 8:
            # bools are a special case
            if not is_boolean:
                msg = f"8 bit ints are not currently supported."
                raise CompilerError(msg)
            return tr.BoolType
        elif bits == 32:
            return tr.Predicate32 if is_boolean else tr.Int32
        elif bits == 64:
            return tr.Predicate64 if is_boolean else tr.Int64
        else:
            msg = f"Unsupported integer bit width {bits}"
            raise CompilerError(msg)
    elif bits == 32:
        return tr.FPredicate32 if is_boolean else tr.Float32
    elif bits == 64:
        return tr.FPredicate64 if is_boolean else tr.Float64
    else:
        msg = f"Unsupported floating point bit width {bits}"
        raise CompilerError(msg)


def array_arg_from_spec(ndims, dtype, fixed_dims=(), evol=None):
    """
    Parameterized array type suitable for use as an argument.
    evol can be None, sliding window, and iterated (just advance iterator by one each time),
    with any subscript applied to a sliding window being folded into the variable's evolution.

    dims should be a dense map, tuple of key, value pairs

    """
    dtype = scalar_type_from_numpy_type(dtype)
    # should be a tuple of pairs
    seen = set()
    for index, value in fixed_dims:
        if index in seen:
            msg = f"index {index} is duplicated."
            raise CompilerError(msg)
        seen.add(index)
        if not isinstance(index, numbers.Integral):
            msg = f"dims can only be used to specify fixed dimensions, received: {dim}."
            raise CompilerError(msg)
        elif 0 > dim:
            msg = f"Negative dim {dim} specified"
            raise CompilerError(msg)
        elif dim >= ndims:
            msg = f"dim {dim} specified for array with {ndims} dimensions."
            raise CompilerError(msg)
    dims = tuple(d for d in fixed_dims)
    return ir.ArrayArg(ndims, dtype, dims, evol)
