from __future__ import annotations

import typing
from dataclasses import dataclass

import ir


@dataclass(frozen=True)
class IntType:
    bitwidth: int
    is_array: typing.ClassVar[bool] = False
    is_integral: typing.ClassVar[bool] = True


@dataclass(frozen=True)
class FloatType:
    bitwidth: int
    is_array: typing.ClassVar[bool] = False
    is_integral: typing.ClassVar[bool] = False


# unfortunately this has to be tracked for lowering to C, where
# intrinsics must sometimes explicitly cast a predicate.

@dataclass(frozen=True)
class PredicateType:
    bitwidth: int
    is_array: typing.ClassVar[bool] = False
    # Treat predicates as integral by default, since they may be subjected to bit manipulation.
    is_integral: typing.ClassVar[bool] = True


@dataclass(frozen=True)
class ArrayType:
    """
    Array type descriptor.

    dtype: scalar type used by this array
    dims: tuple(str or int,...)
    stride: input stride

    """
    dims: typing.Tuple[typing.Union[str, int, ir.NameRef, ir.IntNode], ...]
    dtype: typing.Union[type, IntType, FloatType, PredicateType]
    stride: typing.Optional[typing.Union[int, ir.IntNode]] = None  # inter procedural stride, used for packeting
    is_array: typing.ClassVar[bool] = False
    is_view: typing.ClassVar[bool] = False

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

    base: typing.Union[ArrayType, ViewType]
    subscript: ir.Slice
    is_array: typing.ClassVar[bool] = False
    is_view: typing.ClassVar[bool] = True

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
