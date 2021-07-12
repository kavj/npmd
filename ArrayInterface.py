import typing
from dataclasses import dataclass
from abc import ABC, abstractmethod

import ir


class ArrayBase(ABC):

    @abstractmethod
    def remap_dtype(self, dtype):
        raise NotImplementedError


@dataclass(frozen=True)
class UniformArrayInput(ArrayBase):
    """
    Array type descriptor.

    dtype: scalar type used by this array
    dims: tuple(str or int,...)
    stride: input stride

    """
    dims: typing.Tuple[typing.Union[str, int], ...]
    dtype: type

    def __post_init__(self):
        # These become too ambiguous if not caught here
        if not isinstance(self.dims, tuple):
            msg = f"Type construction expects a tuple of dim arguments, received {type(self.dims)}"
            raise TypeError(msg)
        if not isinstance(self.dtype, type):
            if not isinstance(self.dtype, ir.ScalarType):
                if isinstance(self.dtype, str):
                    # sufficiently common mistake
                    msg = f"Expected dtype argument to be a data type. Received the following string instead: {self.dtype}. " \
                          f"Parsing type information from text is not supported."
                else:
                    msg = f"Expected dtype argument to be an actual type. Received {type(self.dtype)}."
                raise TypeError(msg)

    def remap_dtype(self, dtype):
        return UniformArrayInput(self.dims, dtype)


@dataclass(frozen=True)
class SlidingWindowInput(ArrayBase):
    dims: typing.Tuple[typing.Union[str, int], ...]
    dtype: typing.Union[type, ir.ScalarType]
    stride: int

    def __post_init__(self):
        # These become too ambiguous if not caught here
        if not isinstance(self.dims, tuple):
            msg = f"Type construction expects a tuple of dim arguments, received {type(self.dims)}"
            raise TypeError(msg)
        if not isinstance(self.dtype, type):
            # this is okay if it's an internal type
            if not isinstance(self.dtype, ir.ScalarType):
                if isinstance(self.dtype, str):
                    # sufficiently common mistake
                    msg = f"Expected dtype argument to be a data type. Received the following " \
                      f"string instead: {self.dtype}. Parsing type information from text is not supported."
                else:
                    msg = f"Expected dtype argument to be an actual type. Received {type(self.dtype)}."
                raise TypeError(msg)

    def remap_dtype(self, dtype):
        return SlidingWindowInput(self.dims, dtype, self.stride)


@dataclass(frozen=True)
class ByDimArrayInput(ArrayBase):
    """
    Iterates acrosss consecutive calls over the leading array dim.

    """
    dims: typing.Tuple[typing.Union[str, int, ir.IntNode], ...]
    dtype: type
    stride: int

    def __post_init__(self):
        # These become too ambiguous if not caught here
        if not isinstance(self.dims, tuple):
            msg = f"Type construction expects a tuple of dim arguments, received {type(self.dims)}"
            raise TypeError(msg)
        if not isinstance(self.dtype, type):
            if not isinstance(self.dtype, type):
                # this is okay if it's an internal type
                if isinstance(self.dtype, str):
                    # sufficiently common mistake
                    msg = f"Expected dtype argument to be a data type. Received the following " \
                          f"string instead: {self.dtype}. Parsing type information from text is not supported."
                else:
                    msg = f"Expected dtype argument to be an actual type. Received {type(self.dtype)}."
                raise TypeError(msg)

    def remap_dtype(self, dtype):
        return ByDimArrayInput(self.dims, dtype, self.stride)


class FuncDecl:
    def __init__(self, args, return_type):
        self.args = args
        self.return_type = return_type

    def __hash__(self):
        return hash(self.args)
