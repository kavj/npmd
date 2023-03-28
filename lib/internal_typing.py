import itertools
import typing

from dataclasses import dataclass

from lib.errors import CompilerError


class NumericBase:
    """
    Simple numeric base for internal typing.

    Typing conventions vary quite a lot between type systems used by popular packages, making consistent inference
    difficult.

    Note: there isn't a specific bool type here. It doesn't play well between python facilities and vectorization of
    explicitly boolean types is more difficult than coercion from widened expressions.

    """
    nbits: int
    is_signed: bool
    is_integer: bool
    is_float: bool
    is_real: bool

    @property
    def is_unsigned(self):
        return not self.is_signed

    @property
    def is_complex(self):
        return not self.is_real

    @property
    def real_component_size(self):
        return self.nbits if self.is_real else self.nbits // 2


@dataclass(frozen=True)
class IntType:
    nbits: int
    is_signed: bool = True
    is_integer: typing.ClassVar[bool] = True
    is_float: typing.ClassVar[bool] = False
    is_real: typing.ClassVar[bool] = True

    def __post_init__(self):
        assert isinstance(self.nbits, int)


@dataclass(frozen=True)
class FloatType:
    nbits: int
    is_real: bool = True
    is_signed: typing.ClassVar[bool] = True
    is_integer: typing.ClassVar[bool] = False
    is_float: typing.ClassVar[bool] = True

    def __post_init__(self):
        assert isinstance(self.nbits, int)


int8 = IntType(8)
int16 = IntType(16)
int32 = IntType(32)
int64 = IntType(64)

uint8 = IntType(8, is_signed=False)
uint16 = IntType(16, is_signed=False)
uint32 = IntType(32, is_signed=False)
uint64 = IntType(64, is_signed=False)

float32 = FloatType(32)
float64 = FloatType(64)


complex64 = FloatType(128, is_real=False)
complex128 = FloatType(256, is_real=False)

real_types = (float32, float64)
complex_types = (complex64, complex128)

signed_types = {int8, int16, int32, int64}
unsigned_types = {uint8, uint16, uint32, uint64}

supported_types = {t for t in itertools.chain(complex_types, real_types, signed_types, unsigned_types)}


def is_pow_2(v: int):
    assert isinstance(v, int)
    if v < 2:
         return False
    return v & (v - 1) == 0


def get_type_from_spec(nbits: int, is_integer: bool = True, is_signed: bool = True, is_real: bool = True):
    if is_integer:
        assert is_real
        return IntType(nbits, is_signed)
    else:
        assert is_signed
        return FloatType(nbits, is_real)


def get_max_width(*args):
    """
    Comparison has to return something that behaves like an integer type

    :param args:
    :return:
    """

    return max(a.nbits for a in args)


def get_predicate_type(*args):
    """
    Get a type wide enough to mask the widest operand given
    :param args:
    :return:
    """

    return get_type_from_spec(nbits=get_max_width(args), is_integer=True, is_signed=False, is_real=True)


def promote_to_widest_type(*args):
    """
    Simple type promotion..
    Some notes:

    This tries to match C99 typing rules, except in the case of boolean values.

    This has no concept of an explicit boolean value, because typical conventions of 8-bit booleans complicate coercion
    to wider predicate values.

    :return:
    """

    if any(not a not in supported_types for a in args):
        raise CompilerError(f'Expected only supported NumericBase derived types. Received: {args}')
    nbits = get_max_width(args)
    assert is_pow_2(nbits)

    if all(a.is_integer for a in args):
        if all(a.is_signed for a in args):
            return get_type_from_spec(nbits, is_integer=True, is_signed=True)
        elif all(a.is_unsigned for a in args):
            return get_type_from_spec(nbits, is_integer=True, is_signed=False)
        else:
            # get max width signed type
            max_width_signed = max(a.nbits for a in args if a.is_signed)
            max_width_unsigned = max(a.nbits for a in args if not a.is_unsigned)
            assert is_pow_2(max_width_unsigned) and is_pow_2(max_width_signed)

            is_signed = max_width_signed > max_width_unsigned
            return get_type_from_spec(nbits, is_integer=True, is_signed=is_signed, is_real=True)
    else:
        # we need to know if all types are signed
        if all(a.is_real for a in args):
            return get_type_from_spec(nbits, is_integer=False, is_signed=True, is_real=True)
        else:
            max_real_size = max(a.real_component_size for a in args)
            return get_type_from_spec(max_real_size, is_integer=False, is_signed=True, is_real=False)
