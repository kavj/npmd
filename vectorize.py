import typing

from dataclasses import dataclass

ops = ['add', 'subtract', 'multiply', 'add', 'fma', 'vmin', 'vmax', 'hmin', 'hmax']


@dataclass(frozen=True)
class simdtype:

    bits: int
    name: str


@dataclass(frozen=True)
class op_signature:
    args: typing.Tuple[typing.Hashable,...]
    return_type: typing.Union[simdtype, type]


class ISA:

    features: typing.Dict[str, op_signature]

