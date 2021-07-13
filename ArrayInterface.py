import typing
from dataclasses import dataclass
from abc import ABC, abstractmethod

import ir


@dataclass(frozen=True)
class ArrayInput:
    """
    Array type descriptor.

    dtype: scalar type used by this array
    dims: tuple(str or int,...)
    stride: input stride

    """
    dims: typing.Tuple[typing.Union[str, int], ...]
    dtype: typing.Union[type, ir.ScalarType]
    stride: typing.Optional[int] = None


class FuncDecl:
    def __init__(self, args, return_type):
        self.args = args
        self.return_type = return_type

    def __hash__(self):
        return hash(self.args)
