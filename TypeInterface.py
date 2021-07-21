from __future__ import annotations

import typing
from dataclasses import dataclass

import ir


@dataclass(frozen=True)
class ScalarType:
    """
    Basic numeric type.
    This uses boolean as a flag to allow declaration of fixed width typed predicates.

    Note: bitwidth is required. This presents an issue with initializations of the form "value = 0" or similar.
    Since they're commonly assigned this way for both integer and floating point values, we should just assume
    these do not add numeric type constraints and initially record only the assignment.

    """
    integral: bool
    boolean: bool
    bitwidth: int
    is_array: typing.ClassVar[bool] = False
    is_view: typing.ClassVar[bool] = False


@dataclass(frozen=True)
class ArrayInput:
    """
    Array type descriptor.

    dtype: scalar type used by this array
    dims: tuple(str or int,...)
    stride: input stride

    """
    dims: typing.Tuple[typing.Union[str, int, ir.NameRef, ir.IntNode], ...]
    dtype: typing.Union[type, ScalarType]
    stride: typing.Optional[typing.Union[int, ir.IntNode]] = None  # inter procedural stride, used for packeting

    @property
    def ndims(self):
        return len(self.dims)

    def __post_init__(self):
        # Zero dimensions must be treated as a scalar, not an array.
        assert len(self.dims) > 0


@dataclass(frozen=True)
class ViewType:
    """
    This is used to defer generation of unique array parameters for a particular view.

    """

    base: typing.Union[ArrayInput, ViewType]
    subscript: ir.Slice

    @property
    def dtype(self):
        return self.base.dtype

    @property
    def stride(self):
        return self.base.stride


# This could be part of the symbol table, since it requires internal access anyway.

def make_view_type(array_type, subscript, syms, transpose=False):
    """
    This makes a view of a specific type, not a view reference.
    A view reference requires that we bind to a specific array, via a reference to that array.
    Here we only declare parameters, eg a view with slice parameter [::2] to an array with dims ("n","m").

    Note: transpose has to be handled in the view reference itself. Here we want to apply the correct dims.

    This uses the typical convention of:
        array[slice_params].transpose()

    so slices are applied prior to transposition.

    """
    dims = array_type.dims
    if len(dims) == 0:
        msg = f"Over subscripted array_type {array_type}"
        raise ValueError(msg)
    if isinstance(subscript.slice, ir.Slice):
        # make interval
        pass
    else:
        # single index
        dims = dims[1:]

    dtype = array_type.dtype
    if len(dims) == 0:
        view_type = dtype


class FuncDecl:
    def __init__(self, args, return_type):
        self.args = args
        self.return_type = return_type

    def __hash__(self):
        return hash(self.args)
