from __future__ import annotations

import math
import numbers
import operator
import typing

import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass

from errors import CompilerError

supported_builtins = frozenset({'iter', 'range', 'enumerate', 'zip', 'all', 'any', 'max', 'min', 'abs', 'pow',
                                'round', 'reversed'})


@dataclass(frozen=True)
class Position:
    line_begin: int
    line_end: int
    col_begin: int
    col_end: int


clscond = typing.ClassVar[bool]


class TypeBase:
    pass


@dataclass(frozen=True)
class CObjectType(TypeBase):
    is_array: bool


class ScalarType(TypeBase):
    pass


@dataclass(frozen=True)
class IntegerType(ScalarType):
    bits: int


@dataclass(frozen=True)
class UnsignedType(ScalarType):
    bits: int


@dataclass(frozen=True)
class FloatType(ScalarType):
    bits: int


@dataclass(frozen=True)
class PredicateType(ScalarType):
    bits: int


class StmtBase:
    pos = None


@dataclass(frozen=True)
class VarDecl(StmtBase):
    """
    Variable declaration without explicit assignment.
    This is a visibility hint for code gen to address a few edge cases, such as
    variables that are bound on both sides of an "if else" branch statement, yet
    possibly unbound before it.
    """
    name: NameRef
    type: typing.Any
    initial_value: typing.Optional[ValueRef]
    pos: Position


Statement = typing.TypeVar('Statement', bound=StmtBase)


class ValueRef(ABC):
    """
    This is the start of a base class for expressions.

    For our purposes, we distinguish an expression from a value by asking whether it can be
    an assignment target.

    The term 'expression_like' is used elsewhere in the code to include mutable targets, where
    assignment alters a data structure as opposed to simply binding a value to a name.

    """

    constant: clscond = False

    @property
    def as_constant(self):
        return self


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


class Constant(ValueRef):
    """
    Base class for anything that can be the target of an assignment. 

    """
    value: clscond = None
    constant: clscond = True

    def __init__(self, value):
        if isinstance(value, (bool, np.bool_)):
            value = np.bool_(value)
        elif isinstance(value, numbers.Integral):
            # check if this fits a default integer
            info = np.iinfo(np.int_)
            if info.min <= value <= info.max:
                value = np.int_(value)
            else:
                info = np.iinfo(np.int64)
                if info.min <= value <= info.max:
                    value = np.int64(value)
                else:
                    msg = f"No available fixed width type can hold constant {value}."
                    raise CompilerError(msg)
        elif isinstance(value, numbers.Real):
            if np.isnan(value):
                object.__setattr__(self, 'value', np.nan)
                return
            info = np.finfo(np.float_)
            if info.min <= value <= info.max:
                value = np.float_(value)
            else:
                info = np.finfo(np.float64)
                if info.min <= value <= info.max:
                    value = np.float64(value)
                else:
                    msg = f"No available fixed width type can hold constant {value}."
                    raise CompilerError(msg)
        else:
            msg = f"Only real valued numbers are currently supported, received constant {value}."
            raise CompilerError(msg)
        object.__setattr__(self, 'value', value)

    @property
    def is_predicate(self):
        return isinstance(self.value, np.bool_)

    @property
    def is_integer(self):
        return not self.is_predicate and isinstance(self.value, numbers.Integral)

    def __bool__(self):
        return operator.truth(self.value)


@dataclass(frozen=True)
class StringConst:
    value: str

    def __post_init__(self):
        assert isinstance(self.value, str)
        if any(ord(v) > 127 for v in self.value):
            msg = f"Only strings that can be converted to ascii text are supported. This is mainly intended" \
                  f"to facilitate simple printing support at some point."
            raise CompilerError(msg)


# commonly used

Zero = Constant(0)
One = Constant(1)
Neg_One = Constant(-1)
NAN = Constant(math.nan)
TRUE = Constant(True)
FALSE = Constant(False)


# Top Level

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
            msg = f"No method to make name reference from type {type(name)}."
            raise TypeError(msg)
        object.__setattr__(self, 'name', name)

    def __post_init__(self):
        assert isinstance(self.name, str)


# Todo: specialize for small fixed size arrays

@dataclass(frozen=True)
class ArrayType(TypeBase):
    ndims: int
    dtype: ScalarType

    def __post_init__(self):
        assert isinstance(self.ndims, int)
        assert self.ndims > 0
        assert isinstance(self.dtype, ScalarType)


class ArrayInitializer(Expression):
    shape: Tuple
    dtype: ScalarType

    @property
    def subexprs(self):
        yield self.shape
        yield self.dtype


@dataclass(frozen=True)
class Ones(ArrayInitializer):
    shape: Tuple
    dtype: ScalarType

    def __post_init__(self):
        assert self.shape is None or isinstance(self.shape, tuple)
        assert isinstance(self.dtype, ScalarType)
        if len(self.shape) > 4:
            msg = f"Arrays with more than 4 dims are unsupported here"
            raise CompilerError(msg)


@dataclass(frozen=True)
class Zeros(ArrayInitializer):
    shape: Tuple
    dtype: ScalarType

    def __post_init__(self):
        assert self.shape is None or isinstance(self.shape, tuple)
        assert isinstance(self.dtype, ScalarType)
        if len(self.shape) > 4:
            msg = f"Arrays with more than 4 dims are unsupported here"
            raise CompilerError(msg)


@dataclass(frozen=True)
class Empty(ArrayInitializer):
    shape: Tuple
    dtype: ScalarType

    def __post_init__(self):
        assert self.shape is None or isinstance(self.shape, tuple)
        assert isinstance(self.dtype, ScalarType)
        if len(self.shape) > 4:
            msg = f"Arrays with more than 4 dims are unsupported here"
            raise CompilerError(msg)


@dataclass(frozen=True)
class ArrayArg(Expression):
    ndims: Constant
    dtype: ScalarType

    @property
    def subexprs(self):
        yield self.ndims
        yield self.dtype


@dataclass(frozen=True)
class ArrayRef(ValueRef):
    name: NameRef
    base: ValueRef

    # Todo: redo int check

    def __post_init__(self):
        assert isinstance(self.name, NameRef)


@dataclass(frozen=True)
class ScalarRef(ValueRef):
    name: NameRef
    type: ScalarType


@dataclass(frozen=True)
class SingleDimRef(Expression):
    base: ValueRef
    dim: Constant

    def __post_init__(self):
        if not isinstance(self.dim, Constant) or not self.dim.is_integer:
            msg = f"Expected integer constant, received {self.dim} of type {type(self.dim)}."
            raise TypeError(msg)

    @property
    def subexprs(self):
        yield self.base
        yield self.dim


@dataclass(frozen=True)
class SlidingWindowViewRef:
    """
    This partially implements numpy's sliding window view.
    It has to be part of the IR to avoid inference here.

    This may be added to ir once it's stable...


    Consider the following

    Here we can't packetize edges, as this may not have uniform width.
    If n > len(a) - width + 1, then some iterations are truncated.

    def example(a, width, n, output):
       for i in range(n):
           output[i] = f(a[i:i+width])

    Here it's actually decidable that we have uniform iteration width,
    because len must be non-negative and width < 1 is treated as an error
    to simplify runtime logic. Note that the iteration range would be wrong
    even for zero.

    def example2(a, width, output):
       for i in range(len(a)-width+1):
           output[i] = f(a[i:i+width])

    """

    expr: ValueRef
    base: ValueRef
    width: ValueRef
    stride: ValueRef
    iterated: typing.Optional[bool] = True

    def __post_init__(self):
        assert isinstance(self.base, ValueRef)


@dataclass(frozen=True)
class Subscript(Expression):
    value: ValueRef
    index: ValueRef

    def __post_init__(self):
        if not isinstance(self.value, ValueRef):
            msg = f"Expected ValueRef, got {self.value}, type: {type(self.value)}."
            raise TypeError(msg)
        elif not isinstance(self.index, ValueRef):
            msg = f"Expected ValueRef, got {self.index}, type: {type(self.index)}."
            raise TypeError(msg)

    @property
    def subexprs(self):
        yield self.value
        yield self.index


@dataclass(frozen=True)
class Min(Expression):
    left: ValueRef
    right: ValueRef

    def __init__(self, left, right):
        assert isinstance(left, ValueRef)
        assert isinstance(right, ValueRef)
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)

    @property
    def subexprs(self):
        yield self.left
        yield self.right


@dataclass(frozen=True)
class MinReduction(Expression):
    values: typing.Any

    def __init__(self, *values):
        object.__setattr__(self, "values", frozenset(values))

    @property
    def subexprs(self):
        for v in self.values:
            yield v


@dataclass(frozen=True)
class Max(Expression):
    left: ValueRef
    right: ValueRef

    def __init__(self, left, right):
        assert isinstance(left, ValueRef)
        assert isinstance(right, ValueRef)
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)

    @property
    def subexprs(self):
        yield self.left
        yield self.right


@dataclass(frozen=True)
class MaxReduction(Expression):
    values: typing.Any

    def __init__(self, *values):
        object.__setattr__(self, "values", frozenset(values))

    @property
    def subexprs(self):
        for v in self.values:
            yield v


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
    imports: typing.List[typing.Any]

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

    start: typing.Union[NameRef, Constant]
    stop: typing.Optional[typing.Union[NameRef, Constant]]
    step: typing.Union[NameRef, Constant]

    def __post_init__(self):
        assert isinstance(self.start, (NameRef, Constant))
        assert isinstance(self.stop, (NameRef, Constant))
        assert isinstance(self.step, (NameRef, Constant))

    @property
    def subexprs(self):
        yield self.start
        yield self.stop
        yield self.step


@dataclass(frozen=True)
class Tuple(Expression):
    elements: typing.Tuple[ValueRef, ...]

    def __init__(self, *elements):
        for e in elements:
            if not isinstance(e, ValueRef):
                msg = f"ir tuple has non-IR element: {e}."
                raise TypeError(msg)
        object.__setattr__(self, 'elements', tuple(elements))

    def __post_init__(self):
        for e in self.elements:
            if not isinstance(e, ValueRef):
                msg = f"Expected value refs, received {e}."
                raise TypeError(msg)

    @property
    def subexprs(self):
        for e in self.elements:
            yield e

    @property
    def length(self):
        return len(self.elements)


Targetable = typing.TypeVar('Targetable', NameRef, Subscript, Tuple)


@dataclass(frozen=True)
class Sqrt(Expression):
    operand: ValueRef


class BinOp(Expression):
    # This is just to quell static analyzers

    left: ValueRef
    right: ValueRef

    def __init__(self, left, right):
        raise NotImplementedError("Binops cannot be instantiated.")

    @property
    def subexprs(self):
        yield self.left
        yield self.right


@dataclass(frozen=True)
class ADD(BinOp):
    left: ValueRef
    right: ValueRef


@dataclass(frozen=True)
class SUB(BinOp):
    left: ValueRef
    right: ValueRef


@dataclass(frozen=True)
class MULT(BinOp):
    left: ValueRef
    right: ValueRef


@dataclass(frozen=True)
class TRUEDIV(BinOp):
    left: ValueRef
    right: ValueRef


@dataclass(frozen=True)
class FLOORDIV(BinOp):
    left: ValueRef
    right: ValueRef


@dataclass(frozen=True)
class MOD(BinOp):
    left: ValueRef
    right: ValueRef


@dataclass(frozen=True)
class POW(BinOp):
    left: ValueRef
    right: ValueRef


@dataclass(frozen=True)
class LSHIFT(BinOp):
    left: ValueRef
    right: ValueRef


@dataclass(frozen=True)
class RSHIFT(BinOp):
    left: ValueRef
    right: ValueRef


@dataclass(frozen=True)
class BITOR(BinOp):
    left: ValueRef
    right: ValueRef


@dataclass(frozen=True)
class BITXOR(BinOp):
    left: ValueRef
    right: ValueRef


@dataclass(frozen=True)
class BITAND(BinOp):
    left: ValueRef
    right: ValueRef


@dataclass(frozen=True)
class MATMULT(BinOp):
    left: ValueRef
    right: ValueRef


@dataclass(frozen=True)
class Length(Expression):
    operand: ValueRef

    @property
    def subexprs(self):
        yield self.operand


# Compare ops are once again their own class,
# since they cannot be in place like binops


class CompareOp(Expression):
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


@dataclass(frozen=True)
class NE(CompareOp):
    left: ValueRef
    right: ValueRef


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


@dataclass(frozen=True)
class IN(CompareOp):
    operand: ValueRef


@dataclass(frozen=True)
class NOTIN(CompareOp):
    operand: ValueRef


class BoolOp(Expression):
    """
    Boolean operation using a single logical operation and an arbitrary
    number of operands. Base class is used here to aggregate type checks.

    """
    operands: typing.Iterable[ValueRef, ...]

    @property
    def subexprs(self):
        for operand in self.operands:
            yield operand


@dataclass(frozen=True)
class OR(BoolOp):
    """
    Boolean OR
    """
    operands: typing.Iterable[ValueRef, ...]

    def __init__(self, *operands):
        object.__setattr__(self, 'operands', frozenset(operands))


@dataclass(frozen=True)
class AND(BoolOp):
    """
    Boolean AND
    """
    operands: typing.Tuple[ValueRef, ...]

    def __init__(self, *operands):
        object.__setattr__(self, 'operands', operands)


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


@dataclass(frozen=True)
class Call(Expression):
    """
    An arbitrary call node. This can be replaced
    in cases where it matches an optimizable built in.

    """

    def __init__(self, *args):
        func = args[0]
        args_ = tuple(args[1:])
        object.__setattr__(self, "func", func)
        object.__setattr__(self, "args", args_)

    func: NameRef
    # no support for keyword arguments, as these complicate transforms
    args: typing.Optional[typing.Tuple[ValueRef, ...]]

    @property
    def subexprs(self):
        yield self.func
        for arg in self.args:
            yield arg


@dataclass(frozen=True)
class AffineSeq(Expression):
    """
    This captures range, enumerate, and some generated access functions.
    """
    start: ValueRef
    stop: typing.Optional[ValueRef]
    step: ValueRef

    def __post_init__(self):
        assert isinstance(self.start, ValueRef)
        assert self.stop is None or isinstance(self.stop, ValueRef)
        assert isinstance(self.step, ValueRef)

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
class Select(Expression):
    """
    A Python if expression.

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


class UnaryOp(Expression):
    operand: ValueRef

    @property
    def subexprs(self):
        yield self.operand


@dataclass(frozen=True)
class USUB(UnaryOp):
    operand: ValueRef


@dataclass(frozen=True)
class UNOT(UnaryOp):
    operand: ValueRef


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
            pass
        else:
            raise CompilerError(f"Enumerate takes 1 or 2 arguments, {nargs} given.")
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
        object.__setattr__(self, 'elements', operands)

    elements: typing.Tuple[ValueRef, ...]

    @property
    def subexprs(self):
        for e in self.elements:
            yield e


@dataclass
class Assign(StmtBase):
    """
    An assignment of a right hand side expression to a name or subscripted name.

    """

    target: Targetable
    value: ValueRef
    pos: Position

    def __post_init__(self):
        assert isinstance(self.target, ValueRef)
        assert isinstance(self.value, ValueRef)


@dataclass
class SingleExpr(StmtBase):
    expr: ValueRef
    pos: Position

    def __post_init__(self):
        if not isinstance(self.expr, ValueRef):
            print("something")
        assert isinstance(self.expr, ValueRef)


@dataclass
class Break(StmtBase):
    pos: Position


@dataclass
class Continue(StmtBase):
    pos: Position


@dataclass
class ForLoop(StmtBase):
    target: ValueRef
    iterable: ValueRef
    body: typing.List[Statement]
    pos: Position

    def __post_init__(self):
        if not isinstance(self.target, ValueRef):
            msg = f"Expected ValueRef, got {self.target}, type: {type(self.target)}."
            raise TypeError(msg)
        elif not isinstance(self.iterable, ValueRef):
            msg = f"Expected ValueRef, got {self.iterable}, type: {type(self.iterable)}."
            raise TypeError(msg)


@dataclass
class IfElse(StmtBase):
    test: ValueRef
    if_branch: typing.List[Statement]
    else_branch: typing.List[Statement]
    pos: Position

    def __post_init__(self):
        assert isinstance(self.test, ValueRef)


@dataclass(frozen=True)
class ImportRef:
    module: str
    member: typing.Optional[str]
    alias: typing.Optional[str]
    pos: Position

    @property
    def name(self):
        if self.alias is not None:
            return self.alias
        elif self.member is not None:
            return self.member
        else:
            return self.module

    @property
    def is_module_import(self):
        return self.member is None


@dataclass
class ModImport(StmtBase):
    module: NameRef
    as_name: NameRef
    pos: Position


@dataclass
class NameImport(StmtBase):
    module: NameRef
    name: NameRef
    as_name: NameRef
    pos: Position

    def __post_init__(self):
        assert isinstance(self.module, NameRef)
        assert isinstance(self.name, NameRef)
        assert isinstance(self.as_name, NameRef)


@dataclass
class Return(StmtBase):
    value: typing.Optional[ValueRef]
    pos: Position


@dataclass
class WhileLoop(StmtBase):
    test: ValueRef
    body: typing.List[Statement]
    pos: Position

    def __post_init__(self):
        assert isinstance(self.test, ValueRef)


@dataclass(frozen=True)
class gather(Expression):
    array: ValueRef
    stride: ValueRef
    count: Constant

    def subexprs(self):
        yield self.array
        yield self.stride


compare_ops = {EQ: "==",
               NE: "!=",
               LT: "<",
               LE: "<=",
               GT: ">",
               GE: ">=",
               IN: "in",
               NOTIN: "not in"}

binop_ops = {ADD: "+",
             SUB: "-",
             MULT: "*",
             TRUEDIV: "/",
             FLOORDIV: "//",
             POW: "**"}

unary_ops = {
    USUB: "-",
    UNOT: "~"
}

# partial interning of supported types

Float32 = FloatType(bits=32)
Float64 = FloatType(bits=64)
UInt32 = UnsignedType(bits=32)
UInt64 = UnsignedType(bits=64)
Int32 = IntegerType(bits=32)
Int64 = IntegerType(bits=64)
Predicate32 = PredicateType(bits=32)
Predicate64 = PredicateType(bits=64)
# awkward exception here due to the canonical "bool" type being castable to a signed char
BoolType = PredicateType(bits=8)

supported_types = {Float32, Float64, UInt32, UInt64, Predicate32, Predicate64, BoolType}

# dtypes are returned by promote_types and other parts of the numpy interface
# so we need to include them

by_input_dtype = {np.dtype(np.int32): Int32,
                  np.dtype(np.int64): Int64,
                  np.dtype(np.float32): Float32,
                  np.dtype(np.float64): Float64,
                  np.dtype(np.bool_): BoolType}

# constructable

constr_by_dtype = {
    np.dtype(np.int32): np.int32,
    np.dtype(np.int64): np.int64,
    np.dtype(np.float32): np.float32,
    np.dtype(np.float64): np.float64,
    np.dtype(np.bool_): np.bool_
}

default_int_dtype = by_input_dtype[np.dtype(np.int_)]
default_float_dtype = by_input_dtype[np.dtype(np.float_)]

by_ir_type = {Int32: np.dtype(np.int32),
              Int64: np.dtype(np.int64),
              Float32: np.dtype(np.int64),
              Float64: np.dtype(np.int64),
              BoolType: np.bool_}

np_func_by_binop = {
    ADD: np.add,
    SUB: np.subtract,
    MULT: np.multiply,
    TRUEDIV: np.true_divide,
    FLOORDIV: np.floor_divide,
    MOD: np.mod,
    POW: np.power,
    LSHIFT: np.left_shift,
    RSHIFT: np.right_shift,
    BITOR: np.bitwise_or,
    BITAND: np.bitwise_and,
    BITXOR: np.bitwise_xor,
    UNOT: np.bitwise_not
}

truth_type_table = {
    Int32: Predicate32,
    Int64: Predicate64,
    Float32: Predicate32,
    Float64: Predicate64,
    BoolType: BoolType,
    Predicate32: Predicate32,
    Predicate64: Predicate64
}
