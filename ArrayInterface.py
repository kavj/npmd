import typing
from dataclasses import dataclass

import ir


class ArrayBase:
    pass


@dataclass(frozen=True)
class UniformArrayInput(ArrayBase):
    """
    Array type descriptor.

    dtype: scalar type used by this array
    dims: tuple(str or int,...)
    stride: input stride

    """
    dtype: type
    dims: typing.Tuple[typing.Union[str, int, ir.IntNode], ...]


class SlidingWindowInput(ArrayBase):
    def __init__(self, dtype, dims, stride):
        self.dtype = dtype
        self.dims = dims
        self.stride = stride


class ByDimArrayInput(ArrayBase):
    """
    Iterates acrosss consecutive calls over the leading array dim.

    """

    def __init__(self, dtype, dims):
        self.dtype = dtype
        self.dims = dims
