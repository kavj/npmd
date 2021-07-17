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
    stride: typing.Optional[typing.Union[int, ir.IntNode]] = None
    is_array: typing.ClassVar[bool] = True
    is_view: typing.ClassVar[bool] = False


@dataclass(frozen=True)
class ArrayView:
    base: ir.NameRef
    subscript: ir.Subscript
    is_array: typing.ClassVar[bool] = True
    is_view: typing.ClassVar[bool] = True


class FuncDecl:
    def __init__(self, args, return_type):
        self.args = args
        self.return_type = return_type

    def __hash__(self):
        return hash(self.args)
