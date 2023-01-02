from __future__ import annotations

import math
import numbers
import numpy
import operator
import typing

from abc import ABC, abstractmethod
from dataclasses import dataclass

from lib.errors import CompilerError

supported_builtins = frozenset({'iter', 'range', 'enumerate', 'zip', 'all', 'any', 'max', 'min', 'abs', 'pow',
                                'round', 'reversed'})

int8 = numpy.dtype('int8')
int16 = numpy.dtype('int16')
int32 = numpy.dtype('int32')
int64 = numpy.dtype('int64')
uint8 = numpy.dtype('uint8')
uint16 = numpy.dtype('uint16')
uint32 = numpy.dtype('uint32')
uint64 = numpy.dtype('uint64')
float16 = numpy.dtype('float16')
float32 = numpy.dtype('float32')
float64 = numpy.dtype('float64')
float128 = numpy.dtype('float128')
complex64 = numpy.dtype('complex64')
complex128 = numpy.dtype('complex128')
complex256 = numpy.dtype('complex256')
bool_type = numpy.dtype('bool')


@dataclass(frozen=True, order=True)
class Position:
    line_begin: int
    line_end: int
    col_begin: int
    col_end: int


clscond = typing.ClassVar[bool]


class TypeBase:
    pass


class StmtBase:
    pos = None


Statement = typing.TypeVar('Statement', bound=StmtBase)
StatementList = typing.Union[typing.List[Statement], typing.Tuple[Statement]]


class ValueRef(ABC):
    """
    This is the start of a base class for expressions.

    For our purposes, we distinguish an expression from a value by asking whether it can be
    an assignment target.

    The term 'expression_like' is used elsewhere in the code to include mutable targets, where
    assignment alters a data structure as opposed to simply binding a value to a name.

    """

    pass


class Expression(ValueRef):
    """

    Base class for expression protocol. The rules are:
     * expressions must be immutable and safely hashable
     * expressions must yield their subexpression components as a property of the class
       in a manner suitable to reconstruct the original expression, meaning for expression type E,
       E(*E.subexprs) == E)

     -- This is used by the expression transformer. Without that, expression transformers would need to know
        the exact signature of any expression type it operates on.

    """

    def __post_init__(self):
        assert all(isinstance(e, ValueRef) for e in self.subexprs)

    @property
    @abstractmethod
    def subexprs(self):
        raise NotImplementedError

    def reconstruct(self, *args):
        """
        positional argument only method to reconstruct expression
        :param args:
        :return:
        """
        # this is an instanced method, because we sometimes need to capture
        # information that can't be transferred through sub-expressions
        return self.__class__(*args)


@dataclass(frozen=True)
class CONSTANT(ValueRef):
    """
    ir type for numeric constant
    """
    value: numpy.number
    is_predicate: bool
    bool_dtypes: typing.ClassVar[typing.Set[numpy.dtype]] = {numpy.dtype('bool')}
    int_dtypes: typing.ClassVar[typing.Set[numpy.dtype]] = {int8, int16, int32, int64, uint8, uint16, uint32, uint64}
    float_dtypes: typing.ClassVar[typing.Set[numpy.dtype]] = {float16, float32, float64, float128}
    complex_dtypes: typing.ClassVar[typing.Set[numpy.dtype]] = {complex64, complex128, complex256}

    def __bool__(self):
        return operator.truth(self.value)

    def __post_init__(self):
        assert hasattr(self.value, 'dtype') or numpy.isnan(self.value)

    def can_negate(self):
        if self.is_integer:
            if not (-2 ** 63 <= (-self.value) <= 2 ** 63 - 1):
                return False
        return True

    def matches(self, other: CONSTANT):
        if self.is_nan:
            return other.is_nan
        return self.value == other.value and self.is_predicate == other.is_predicate

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def negative(self):
        return self.value < 0

    @property
    def is_nan(self):
        return numpy.isnan(self.value)

    @property
    def is_bool(self):
        return self.dtype in self.bool_dtypes

    @property
    def is_integer(self):
        return self.dtype in self.int_dtypes

    @property
    def is_float(self):
        return self.dtype in self.float_dtypes


def is_nan(value: ValueRef):
    assert isinstance(value, ValueRef)
    return isinstance(value, CONSTANT) and numpy.isnan(value.value)


@dataclass(frozen=True)
class StringConst(ValueRef):
    value: str

    def __post_init__(self):
        assert isinstance(self.value, str)
        if any(ord(v) > 127 for v in self.value):
            msg = f"Only strings that can be converted to ascii text are supported. This is mainly intended" \
                  f"to facilitate simple printing support at some point."
            raise CompilerError(msg)

    def __bool__(self):
        return operator.truth(self.value)


def wrap_constant(c: typing.Union[str, bool, numpy.bool_, numbers.Number], dtype: typing.Optional[numpy.dtype] = None, is_predicate: bool = False):
    # check if we have a supported type
    if isinstance(c, str):
        value = StringConst(c)
    elif math.isnan(c):
        # numpy uses a different nan object than math.nan
        # but numpy.isnan is somewhat too strict as we don't want type coercion to float. We only need to know
        # that the input was not a floating point nan.
        value = CONSTANT(numpy.nan, is_predicate)
    elif isinstance(c, (bool, numpy.bool_)):
        # ensure we wrap the numpy bool_ type
        if dtype is None:
            c = numpy.bool_(c)
        else:
            c = dtype.type(c)
        value = CONSTANT(c, is_predicate)
    elif dtype is None:
        assert isinstance(c, numbers.Number)
        min_scalar_type: numpy.dtype = numpy.min_scalar_type(c)
        if min_scalar_type == numpy.dtype('O'):
            msg = f'Cannot map "{c}" to non-object type'
            raise CompilerError(msg)
        raw_value = min_scalar_type.type(c)
        value = CONSTANT(raw_value, is_predicate)
    else:
        raw_value = dtype.type(c)
        value = CONSTANT(raw_value, is_predicate)
    return value


# commonly used
Half = wrap_constant(0.5)
Two = wrap_constant(2)
One = wrap_constant(1)
Zero = wrap_constant(0)
NegativeOne = wrap_constant(-1)
NegativeTwo = wrap_constant(-2)
TRUE = wrap_constant(True)
FALSE = wrap_constant(False)
NAN = wrap_constant(numpy.nan)


def integer_cast(node: ValueRef):
    while isinstance(node, CAST):
        node = node.value
    return CAST(node, numpy.dtype('int64'))


def float_cast(node: ValueRef):
    while isinstance(node, CAST):
        node = node.value
    return CAST(node, numpy.dtype('float64'))


# Top Level

@dataclass(frozen=True)
class NoneRef(ValueRef):
    # sentinel value, since it's b
    pass


# Todo: add typing to these to avoid needing constant access to symbol table
@dataclass(frozen=True)
class NameRef(ValueRef):
    # variable name ref
    name: str

    def __init__(self, name):
        # If we need a name reference from a string or nameref, we can
        # just handle it here so as to avoid extra utility functions.
        if isinstance(name, NameRef):
            name = name.name
        elif not isinstance(name, str):
            msg = f'No method to make name reference from type {type(name)}.'
            raise TypeError(msg)
        object.__setattr__(self, 'name', name)


# Todo: specialize for small fixed size arrays

@dataclass(frozen=True)
class ArrayType(TypeBase):
    ndims: int
    dtype: numpy.dtype

    def __post_init__(self):
        assert isinstance(self.ndims, int)
        assert self.ndims > 0
        assert isinstance(self.dtype, numpy.dtype)

    def without_leading_dim(self):
        if self.ndims == 1:
            return self.dtype
        else:
            return ArrayType(self.ndims - 1, self.dtype)


@dataclass(frozen=True)
class ArrayInitializer(Expression):
    shape: ValueRef
    dtype: StringConst
    fill_value: typing.Union[CONSTANT, NoneRef]

    @property
    def subexprs(self):
        yield self.shape

    def reconstruct(self, *args):
        assert len(args) == 1
        return super().reconstruct(args[0], self.dtype, self.fill_value)

    def __post_init__(self):
        assert isinstance(self.dtype, StringConst)
        if isinstance(self.shape, TUPLE) and len(self.shape) > 4:
            msg = f'Arrays with more than 4 dims are unsupported here'
            raise CompilerError(msg)


@dataclass(frozen=True)
class ArrayFill(StmtBase):
    array: NameRef
    fill_value: CONSTANT
    pos: Position


@dataclass(frozen=True)
class ArrayAlloc(Expression):
    shape: ValueRef
    dtype: StringConst

    @property
    def subexprs(self):
        yield self.shape

    def reconstruct(self, *args):
        assert len(args) == 1
        return ArrayAlloc(args[0], self.dtype)

    def __post_init__(self):
        assert isinstance(self.dtype, StringConst)
        if isinstance(self.shape, TUPLE) and len(self.shape) > 4:
            msg = f'Arrays with more than 4 dims are unsupported here'
            raise CompilerError(msg)


@dataclass(frozen=True)
class ShapeRef(Expression):
    value: typing.Tuple[ValueRef, ...]

    @property
    def subexprs(self):
        yield from self.value


@dataclass(frozen=True)
class SingleDimRef(Expression):
    base: ValueRef
    dim: CONSTANT

    def __post_init__(self):
        if not isinstance(self.dim, CONSTANT) or not isinstance(self.dim.value, numbers.Integral):
            msg = f'Expected integer constant, received {self.dim} of type {type(self.dim)}.'
            raise TypeError(msg)
        elif self.dim.value < 0:
            msg = f'Negative indices are unsupported: "{self.dim}"'
            raise CompilerError(msg)

    @property
    def subexprs(self):
        yield self.base
        yield self.dim


@dataclass(frozen=True)
class ABSOLUTE(Expression):
    value: ValueRef

    @property
    def subexprs(self):
        yield self.value


@dataclass(frozen=True)
class FLOOR(Expression):
    value: ValueRef

    @property
    def subexprs(self):
        yield self.value


@dataclass(frozen=True)
class Subscript(Expression):
    value: NameRef
    index: ValueRef

    def __post_init__(self):
        if not isinstance(self.value, NameRef):
            if isinstance(self.value, Subscript):
                msg = f"Nested subscripts are not currently supported"
                raise TypeError(msg)
            msg = f'Expected name, got {self.value}, type: {type(self.value)}.'
            raise TypeError(msg)
        elif not isinstance(self.index, ValueRef):
            msg = f'Expected ValueRef, got {self.index}, type: {type(self.index)}.'
            raise TypeError(msg)
        elif isinstance(self.index, CONSTANT):
            if self.index.value < 0:
                msg = f'Negative indices are unsupported: "{self.index.value}"'
                raise CompilerError(msg)

    @property
    def subexprs(self):
        yield self.value
        yield self.index


@dataclass(frozen=True)
class MIN(Expression):
    """
    unordered version of min
    """
    left: ValueRef
    right: ValueRef

    def __post_init__(self):
        assert isinstance(self.left, ValueRef)
        assert isinstance(self.right, ValueRef)

    @property
    def subexprs(self):
        yield self.left
        yield self.right


@dataclass(frozen=True)
class MinReduction(Expression):
    values: typing.FrozenSet[ValueRef, ...]

    def __init__(self, *values):
        assert len(values) > 0
        for v in values:
            if not isinstance(v, ValueRef):
                msg = f'Cannot apply min reduction over "{v}"'
                raise TypeError(msg)
        object.__setattr__(self, 'values', frozenset(values))

    @property
    def subexprs(self):
        yield from self.values


@dataclass(frozen=True)
class MAX(Expression):
    left: ValueRef
    right: ValueRef

    @property
    def subexprs(self):
        yield self.left
        yield self.right


@dataclass(frozen=True)
class MAXELT(Expression):
    value: ValueRef

    @property
    def subexprs(self):
        yield self.value


@dataclass(frozen=True)
class MINELT(Expression):
    value: ValueRef

    @property
    def subexprs(self):
        yield self.value


@dataclass(frozen=True)
class MaxReduction(Expression):
    values: typing.FrozenSet[ValueRef, ...]

    def __init__(self, *values):
        assert len(self.values) > 0
        for v in values:
            if not isinstance(v, ValueRef):
                msg = f'Canot apply min reduction over "{v}"'
                raise TypeError(msg)
        object.__setattr__(self, 'values', frozenset(values))

    @property
    def subexprs(self):
        yield from self.values


@dataclass
class Function:
    name: str
    args: typing.List[NameRef]
    body: typing.List[Statement]


@dataclass
class Module:
    """
    This aggregates imports and Function graphs

    """
    name: str
    functions: typing.List[Function]
    module_imports: typing.Dict[str, str]
    name_imports: typing.Dict[str, str]

    def lookup(self, func_name):
        for func in self.functions:
            if func.name == func_name:
                return func


@dataclass(frozen=True)
class Slice(Expression):
    """
    IR representation of a slice.

    Per dimension constraints can be applied

    """

    start: ValueRef
    stop: ValueRef
    step: ValueRef

    def __post_init__(self):
        if self.step == Zero:
            msg = f'Detected a step size of zero in "{self}".'
            raise ValueError(msg)
        for t in (self.start, self.stop, self.step):
            assert isinstance(t, ValueRef)
            if isinstance(t, CONSTANT):
                if t.value < 0:
                    msg = f'Slicing with negative parameters is unsupported'
                    raise CompilerError(msg)

    @property
    def subexprs(self):
        yield self.start
        yield self.stop
        yield self.step


@dataclass(frozen=True)
class TUPLE(Expression):
    elements: typing.Tuple[ValueRef, ...]

    def __init__(self, *elements):
        for e in elements:
            if not isinstance(e, ValueRef):
                msg = f'ir tuple has non-IR element: "{e}".'
                raise TypeError(msg)
        object.__setattr__(self, 'elements', tuple(elements))

    def __post_init__(self):
        for e in self.elements:
            if not isinstance(e, ValueRef):
                msg = f'Expected value refs, received "{e}".'
                raise TypeError(msg)

    def __len__(self):
        return len(self.elements)

    @property
    def subexprs(self):
        for e in self.elements:
            yield e

    @property
    def length(self):
        return len(self.elements)


@dataclass(frozen=True)
class SQRT(Expression):
    operand: ValueRef

    @property
    def subexprs(self):
        yield self.operand


class UnaryOp(Expression):
    operand: ValueRef

    @property
    def subexprs(self):
        yield self.operand


@dataclass(frozen=True)
class USUB(UnaryOp):
    operand: ValueRef


@dataclass(frozen=True)
class UINVERT(UnaryOp):
    operand: ValueRef


class BinOp(Expression):
    # This is just to quell static analyzers

    left: ValueRef
    right: ValueRef
    op: typing.ClassVar[typing.Callable]

    @property
    def subexprs(self):
        yield self.left
        yield self.right


class Contraction(Expression):
    a: ValueRef
    b: ValueRef
    c: ValueRef

    @property
    def subexprs(self):
        yield self.a
        yield self.b
        yield self.c


@dataclass(frozen=True)
class MultiplyAdd(Contraction):
    """
    a * b + c
    """

    a: ValueRef
    b: ValueRef
    c: ValueRef


@dataclass(frozen=True)
class MultiplySub(Contraction):
    """
    a * b - c
    """

    a: ValueRef
    b: ValueRef
    c: ValueRef


@dataclass(frozen=True)
class MultiplyNegateAdd(Contraction):
    """
    -(a * b) + c
    """

    a: ValueRef
    b: ValueRef
    c: ValueRef


@dataclass(frozen=True)
class MultiplyNegateSub(Contraction):
    """
    a * b - c
    """

    a: ValueRef
    b: ValueRef
    c: ValueRef


@dataclass(frozen=True)
class CAST(Expression):
    value: ValueRef
    target_type: TypeBase

    @property
    def subexprs(self):
        yield self.value

    def reconstruct(self, *args):
        # this requires a custom reconstruct method. Since target_type
        # is not a value reference, it cannot be visited as an expression
        assert len(args) == 1
        return super().reconstruct(args[0], self.target_type)


@dataclass(frozen=True)
class ADD(BinOp):
    left: ValueRef
    right: ValueRef

    def __init__(self, *values):
        # use reverse sort to avoid putting constants first
        left, right = sorted(values, key=lambda k: str(k), reverse=True)
        object.__setattr__(self, 'left', left)
        object.__setattr__(self, 'right', right)


@dataclass(frozen=True)
class SUB(BinOp):
    left: ValueRef
    right: ValueRef


@dataclass(frozen=True)
class MULT(BinOp):
    left: ValueRef
    right: ValueRef

    def __init__(self, *values):
        # These are sorted so that
        left, right = sorted(values, key=lambda k: str(k), reverse=True)
        object.__setattr__(self, 'left', left)
        object.__setattr__(self, 'right', right)


@dataclass(frozen=True)
class TRUEDIV(BinOp):
    left: ValueRef
    right: ValueRef

    def __post_init__(self):
        # Catch these early, where it's simplest
        if isinstance(self.right, CONSTANT):
            if self.right == Zero:
                msg = f'Invalid constant operand in true division expression "{self.right}".'
                raise CompilerError(msg)
            elif self.right.is_bool:
                msg = f'Boolean value {self.right} is unsupported for ' \
                      f'true division expression "{self.left} // {self.right}".'
                raise CompilerError(msg)
            elif self.right.is_nan:
                msg = f'Nan value {self.right} is unsupported for true division ' \
                      f'expression "{self.left} // {self.right}".'
                raise CompilerError(msg)


@dataclass(frozen=True)
class FLOORDIV(BinOp):
    left: ValueRef
    right: ValueRef

    def __post_init__(self):
        # Catch these early, where it's simplest
        if isinstance(self.right, CONSTANT):
            if self.right == Zero:
                msg = f'Invalid constant operand in floordiv expression "{self.right}".'
                raise CompilerError(msg)
            elif self.right.is_bool:
                msg = f'Boolean value "{self.right}" is unsupported for ' \
                      f'floor division expression "{self.left} // {self.right}".'
                raise CompilerError(msg)
            elif self.right.is_nan:
                msg = f'Nan value "{self.right}" is unsupported for floor division ' \
                      f'expression "{self.left} // {self.right}".'
                raise CompilerError(msg)


@dataclass(frozen=True)
class MOD(BinOp):
    left: ValueRef
    right: ValueRef

    def __post_init__(self):
        # Catch these early, where it's simplest
        if isinstance(self.right, CONSTANT):
            if self.right == Zero:
                msg = f'Invalid constant operand in modulo expression "{self.right}".'
                raise CompilerError(msg)
            elif self.right.is_bool:
                msg = f'Modulo boolean value "{self.right}" is unsupported.'
                raise CompilerError(msg)
            elif self.right.is_nan:
                msg = f'Modulo nan value "{self.right}" is unsupported.'
                raise CompilerError(msg)


@dataclass(frozen=True)
class POW(BinOp):
    left: ValueRef
    right: ValueRef


@dataclass(frozen=True)
class LSHIFT(BinOp):
    left: ValueRef
    right: ValueRef

    def __post_init__(self):
        if isinstance(self.left, CONSTANT):
            if not self.left.is_integer:
                if self.left.is_bool:
                    msg = f'Coercing bool to integer is not supported in expression "{self.left} << {self.right}".'
                else:
                    msg = f'Expression "{self.left} << {self.right}" is invalid for non-integer operand "{self.left}".'
                raise CompilerError(msg)

        if isinstance(self.right, CONSTANT):
            if not self.right.is_integer:
                if self.right.is_bool:
                    msg = f'Coercing bool to integer is not supported in expression "{self.left} << {self.right}".'
                else:
                    msg = f'Expression "{self.left} << {self.right}" is invalid for non-integer operand {self.right}.'
                raise CompilerError(msg)


@dataclass(frozen=True)
class RSHIFT(BinOp):
    left: ValueRef
    right: ValueRef

    def __post_init__(self):
        if isinstance(self.left, CONSTANT):
            if not self.left.is_integer:
                if self.left.is_bool:
                    msg = f'Coercing bool to integer is not supported in expression "{self.left} >> {self.right}".'
                else:
                    msg = f'Expression "{self.left} << {self.right}" is invalid for non-integer operand "{self.left}".'
                raise CompilerError(msg)

        if isinstance(self.right, CONSTANT):
            if not self.right.is_integer:
                if self.right.is_bool:
                    msg = f'Coercing bool to integer is not supported in expression "{self.left} >> {self.right}".'
                else:
                    msg = f'Expression "{self.left} >> {self.right}" is invalid for non-integer operand "{self.right}".'
                raise CompilerError(msg)


@dataclass(frozen=True)
class BITOR(BinOp):
    left: ValueRef
    right: ValueRef

    def __init__(self, *values):
        # These are sorted so that
        left, right = sorted(values, key=lambda k: str(k))
        object.__setattr__(self, 'left', left)
        object.__setattr__(self, 'right', right)

    def __post_init__(self):
        for term in (self.left, self.right):
            if isinstance(term, CONSTANT) and not term.is_integer:
                msg = f'Unsupported term "{term}" in bitwise expression.'
                raise CompilerError(msg)


@dataclass(frozen=True)
class BITXOR(BinOp):
    left: ValueRef
    right: ValueRef

    def __init__(self, *values):
        # These are sorted so that
        left, right = sorted(values, key=lambda k: str(k), reverse=True)
        object.__setattr__(self, 'left', left)
        object.__setattr__(self, 'right', right)

    def __post_init__(self):
        for term in (self.left, self.right):
            if isinstance(term, CONSTANT) and not term.is_integer:
                msg = f'Unsupported term "{term}" in bitwise expression.'
                raise CompilerError(msg)


@dataclass(frozen=True)
class BITAND(BinOp):
    left: ValueRef
    right: ValueRef

    def __init__(self, *values):
        # These are sorted so that
        left, right = sorted(values, key=lambda k: str(k), reverse=True)
        object.__setattr__(self, 'left', left)
        object.__setattr__(self, 'right', right)

    def __post_init__(self):
        for term in (self.left, self.right):
            if isinstance(term, CONSTANT) and not term.is_integer:
                msg = f'Unsupported term "{term}" in bitwise expression.'
                raise CompilerError(msg)


@dataclass(frozen=True)
class MATMULT(BinOp):
    left: ValueRef
    right: ValueRef


# Compare ops are once again their own class,
# since they cannot be in place like binops


class CompareOp(Expression):
    #  basically an alias to BinOp at this point..
    left: ValueRef
    right: ValueRef

    @property
    def subexprs(self):
        yield self.left
        yield self.right


@dataclass(frozen=True)
class EQ(CompareOp):
    left: ValueRef
    right: ValueRef

    def __init__(self, *values):
        left, right = sorted(values, key=lambda k: str(k), reverse=True)
        object.__setattr__(self, 'left', left)
        object.__setattr__(self, 'right', right)


@dataclass(frozen=True)
class NE(CompareOp):
    left: ValueRef
    right: ValueRef

    def __init__(self, *values):
        left, right = sorted(values, key=lambda k: str(k), reverse=True)
        object.__setattr__(self, 'left', left)
        object.__setattr__(self, 'right', right)


@dataclass(frozen=True)
class LT(CompareOp):
    left: ValueRef
    right: ValueRef


@dataclass(frozen=True)
class LE(CompareOp):
    left: ValueRef
    right: ValueRef


@dataclass(frozen=True)
class GT(CompareOp):
    left: ValueRef
    right: ValueRef


@dataclass(frozen=True)
class GE(CompareOp):
    left: ValueRef
    right: ValueRef


# Todo: These two are not correct. Fix..

@dataclass(frozen=True)
class IN(CompareOp):
    left: ValueRef
    right: ValueRef


@dataclass(frozen=True)
class NOTIN(CompareOp):
    operand: ValueRef
    target: ValueRef


class BoolOp(Expression):
    """
    Boolean operation using a single logical operation and an arbitrary
    number of operands. Base class is used here to aggregate type checks.

    """
    operands: typing.FrozenSet[ValueRef, ...]

    @property
    def subexprs(self):
        yield from self.operands


@dataclass(frozen=True)
class OR(BoolOp):
    """
    Boolean OR
    """
    operands: typing.Tuple[ValueRef, ...]

    def __init__(self, *operands):
        object.__setattr__(self, 'operands', frozenset(operands))


@dataclass(frozen=True)
class AND(BoolOp):
    """
    Boolean AND
    """
    operands: typing.Tuple[ValueRef, ...]

    def __init__(self, *operands):
        object.__setattr__(self, 'operands', frozenset(operands))


@dataclass(frozen=True)
class XOR(BoolOp):
    """
    Boolean XOR
    """
    operands: typing.Tuple[ValueRef, ...]

    def __init__(self, *operands):
        object.__setattr__(self, 'operands', frozenset(operands))


@dataclass(frozen=True)
class TRUTH(BoolOp):
    """
    Truth test for a single operand, which is not known to be wrapped by an implicit truth test.
    This is primarily used to handle cases where folding otherwise leaves an invalid
    expression.

    For example consider an expression that reduces to:

        value = b and non_zero_constant

    This is treated as:

        value = b and True

    After folding the truth constant, we need some way to represent that we export
    the truth test of "b" rather than its actual value, thus

        value = TRUTH(b)

    If instead it's something like:

        if b and True:
            ...

    or:

        value = not (b and non_zero_constant)

    then truth testing is applied by an outer expression anyway.

    """
    operand: ValueRef

    def __post_init__(self):
        assert isinstance(self.operand, ValueRef)

    @property
    def subexprs(self):
        yield self.operand

    @property
    def simplified(self):
        if isinstance(self.operand, CONSTANT):
            return TRUE if operator.truth(self.operand) else FALSE
        return self


@dataclass(frozen=True)
class NOT(BoolOp):
    """
    Boolean not
    """

    operand: ValueRef

    def __post_init__(self):
        assert isinstance(self.operand, ValueRef)

    @property
    def subexprs(self):
        yield self.operand

    @property
    def simplified(self):
        if isinstance(self.operand, NOT):
            if isinstance(self.operand.operand, BoolOp):
                return self.operand.operand
            return TRUTH(self.operand.operand)
        return self


@dataclass(frozen=True)
class PRODUCT(Expression):
    value: ValueRef

    @property
    def subexprs(self):
        yield self.value


@dataclass(frozen=True)
class RECIPROCAL(Expression):
    value: ValueRef

    @property
    def subexprs(self):
        yield self.value


@dataclass(frozen=True)
class SUM(Expression):
    value: ValueRef

    @property
    def subexprs(self):
        yield self.value


@dataclass(frozen=True)
class Call(Expression):
    """
    An arbitrary call node. This can be replaced
    in cases where it matches an optimizable built in.

    """

    func: str
    # no support for keyword arguments, as these complicate transforms
    args: TUPLE
    kwargs: TUPLE

    @property
    def subexprs(self):
        # don't yield call name,
        # since this shouldn't be visited like a variable
        yield self.args
        yield self.kwargs

    def reconstruct(self, *args):
        args_, kwargs = args
        assert len(args_.elements) == len(self.args.elements)
        return Call(self.func, args_, kwargs)


@dataclass(frozen=True)
class AffineSeq(Expression):
    """
    This captures range, enumerate, and some generated access functions.
    """
    start: ValueRef
    # we can't completely avoid the possibility of None here, since it's possible
    # to encounter cases with zip where len() is unsupported, yet enumerate is
    stop: typing.Optional[ValueRef]
    step: ValueRef

    def __post_init__(self):
        if not isinstance(self.start, ValueRef):
            msg = f'Start param of affine sequence must be a value reference. Received: "{self.start}".'
            raise ValueError(msg)
        elif self.stop is not None and not isinstance(self.stop, ValueRef):
            msg = f'Stop param of affine sequence must be a value reference. Received: "{self.stop}".'
            raise ValueError(msg)
        elif not isinstance(self.step, ValueRef):
            msg = f'Step param of affine sequence must be a value reference. Received: "{self.step}".'
            raise ValueError(msg)
        elif self.step == Zero:
            msg = f'Step size of zero detected in "{self}".'
            raise ValueError(msg)

    @property
    def reversed(self):
        raise NotImplementedError

    @property
    def subexprs(self):
        yield self.start
        if self.stop is not None:
            yield self.stop
        yield self.step


@dataclass(frozen=True)
class ALL(Expression):
    value: ValueRef

    @property
    def subexprs(self):
        yield self.value


@dataclass(frozen=True)
class ANY(Expression):
    value: ValueRef

    @property
    def subexprs(self):
        yield self.value


@dataclass(frozen=True)
class SELECT(Expression):
    """
    A Python if-expression.

    This is also being reused for the array and vector cases, since a scalar predicate can just return one
    of 2 vectors.

    This is somewhat awkward coming from the Python side, where we can't assign a truth value to a non-empty array.
    From the IR side, the predicate type can determine how these are blended (scalar, array, or short vector).

    """

    predicate: ValueRef
    on_true: ValueRef
    on_false: ValueRef

    def __post_init__(self):
        assert isinstance(self.predicate, ValueRef)
        assert isinstance(self.on_true, ValueRef)
        assert isinstance(self.on_false, ValueRef)

    @property
    def subexprs(self):
        yield self.predicate
        yield self.on_true
        yield self.on_false


@dataclass(frozen=True)
class ELTWISEOP(Expression):
    a: ValueRef
    b: ValueRef
    out: ValueRef
    where: ValueRef


@dataclass(frozen=True)
class ELTWISEUNARYOP(Expression):
    a: ValueRef
    out: ValueRef
    where: ValueRef


@dataclass(frozen=True)
class LOGICAL_REDUC(Expression):
    a: Expression
    axis: CONSTANT
    out: ValueRef
    keepdims: CONSTANT
    where: ValueRef


@dataclass(frozen=True)
class ARITH_REDUC(Expression):
    a: Expression
    axis: CONSTANT
    out: ValueRef
    keepdims: CONSTANT
    initial: ValueRef
    where: ValueRef


@dataclass(frozen=True)
class Reversed(Expression):
    """
    Sentinel for a "reversed" object

    """

    iterable: ValueRef

    def __post_init__(self):
        assert isinstance(self.iterable, ValueRef)

    @property
    def subexprs(self):
        yield self.iterable


@dataclass(frozen=True)
class Enumerate(Expression):
    """
    High level sentinel, representing an enumerate object.
    """

    iterable: ValueRef
    start: ValueRef

    def __init__(self, *args):
        nargs = len(args)
        if nargs == 1:
            iterable, = args
            start = Zero
        elif nargs == 2:
            iterable, start = args
        else:
            raise CompilerError(f'Enumerate takes 1 or 2 arguments, "{nargs}" given.')
        object.__setattr__(self, 'iterable', iterable)
        object.__setattr__(self, 'start', start)

    @property
    def subexprs(self):
        yield self.iterable
        yield self.start


@dataclass(frozen=True)
class Zip(Expression):
    """
    High level sentinel representing a zip object.
    """

    def __init__(self, *operands):
        assert all(isinstance(e, ValueRef) for e in operands)
        object.__setattr__(self, 'elements', operands)

    elements: typing.Tuple[ValueRef, ...]

    @property
    def subexprs(self):
        for e in self.elements:
            yield e


@dataclass(frozen=True)
class InPlaceOp(StmtBase):
    # Todo: set target explicitly for multiply accum which accumulates to expr.right here
    target: typing.Union[NameRef, Subscript]
    value: BinOp
    pos: Position

    def __post_init__(self):
        # contraction added so that we can represent a += b * c as a contracted op
        assert isinstance(self.value, BinOp)
        assert isinstance(self.target, (NameRef, Subscript))
        assert self.target in self.value.subexprs


@dataclass(frozen=True)
class Assign(StmtBase):
    """
    An assignment of a right hand side expression to a name or subscripted name.

    """

    target: typing.Union[NameRef, Subscript, TUPLE]
    value: ValueRef
    pos: Position

    def __post_init__(self):
        # no support for assigning to arbitary constructs
        if not isinstance(self.target, (NameRef, Subscript)):
            msg = f'Assign expects a name or subsccript target, received "{self.target}"'
            raise TypeError(msg)
        assert isinstance(self.value, ValueRef)
        if isinstance(self.value, Slice):
            msg = f'Explicitly assigning slice expressions is unsupported: "{self}".'
            raise CompilerError(msg)


@dataclass(frozen=True)
class SingleExpr(StmtBase):
    value: ValueRef
    pos: Position

    def __post_init__(self):
        assert isinstance(self.value, ValueRef)


@dataclass(frozen=True)
class Break(StmtBase):
    pos: Position


@dataclass(frozen=True)
class Continue(StmtBase):
    pos: Position


@dataclass
class ForLoop(StmtBase):
    target: typing.Union[NameRef, TUPLE]
    iterable: ValueRef
    body: typing.List[Statement]
    pos: Position

    def __post_init__(self):
        if not isinstance(self.target, (NameRef, TUPLE)):
            msg = f'Expected ValueRef, got "{self.target}", type: "{type(self.target)}".'
            raise TypeError(msg)
        elif not isinstance(self.iterable, ValueRef):
            msg = f'Expected ValueRef, got "{self.iterable}", type: "{type(self.iterable)}".'
            raise TypeError(msg)


@dataclass
class IfElse(StmtBase):
    test: ValueRef
    if_branch: StatementList
    else_branch: StatementList
    pos: Position

    def __post_init__(self):
        assert isinstance(self.test, ValueRef)


@dataclass(frozen=True)
class Return(StmtBase):
    value: typing.Optional[ValueRef]
    pos: Position

    def __post_init__(self):
        assert isinstance(self.value, ValueRef)


@dataclass
class WhileLoop(StmtBase):
    test: ValueRef
    body: typing.List[Statement]
    pos: Position

    def __post_init__(self):
        assert isinstance(self.test, ValueRef)
