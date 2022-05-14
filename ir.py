from __future__ import annotations

import numbers
import operator
import typing

import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass

import ir
from errors import CompilerError

supported_builtins = frozenset({'iter', 'range', 'enumerate', 'zip', 'all', 'any', 'max', 'min', 'abs', 'pow',
                                'round', 'reversed'})

int32 = np.dtype('int32')
int64 = np.dtype('int64')
float32 = np.dtype('float32')
float64 = np.dtype('float64')
bool_type = np.dtype('bool')


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

    @classmethod
    def reconstruct(cls, *args):
        """
        positional argument only method to reconstruct expression
        :param args:
        :return:
        """
        return cls(*args)


@dataclass(frozen=True)
class Constant(ValueRef):
    """
    ir type for numeric constant
    """
    value: np.number
    constant: clscond = True

    @property
    def dtype(self):
        return self.value.dtype

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

    def __bool__(self):
        return operator.truth(self.value)


def wrap_constant(c):
    # check if we have a supported type
    if isinstance(c, str):
        value = StringConst(c)
    else:
        raw_value = np.min_scalar_type(c).type(c)
        value = Constant(raw_value)
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
NAN = wrap_constant(np.nan)


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
    dtype: np.dtype

    def __post_init__(self):
        assert isinstance(self.ndims, int)
        assert self.ndims > 0
        assert isinstance(self.dtype, np.dtype)


class ArrayInitializer(Expression):
    shape: Tuple
    dtype: np.dtype

    @property
    def subexprs(self):
        yield self.shape
        yield self.dtype


@dataclass(frozen=True)
class Ones(ArrayInitializer):
    shape: Tuple
    dtype: np.dtype

    def __post_init__(self):
        assert self.shape is None or isinstance(self.shape, tuple)
        assert isinstance(self.dtype, np.dtype)
        if len(self.shape) > 4:
            msg = f"Arrays with more than 4 dims are unsupported here"
            raise CompilerError(msg)


@dataclass(frozen=True)
class Zeros(ArrayInitializer):
    shape: Tuple
    dtype: np.dtype

    def __post_init__(self):
        assert self.shape is None or isinstance(self.shape, tuple)
        assert isinstance(self.dtype, np.dtype)
        if len(self.shape) > 4:
            msg = f"Arrays with more than 4 dims are unsupported here"
            raise CompilerError(msg)


@dataclass(frozen=True)
class Empty(ArrayInitializer):
    shape: Tuple
    dtype: np.dtype

    def __post_init__(self):
        assert self.shape is None or isinstance(self.shape, tuple)
        assert isinstance(self.dtype, np.dtype)
        if len(self.shape) > 4:
            msg = f"Arrays with more than 4 dims are unsupported here"
            raise CompilerError(msg)


@dataclass(frozen=True)
class ArrayArg(Expression):
    ndims: Constant
    dtype: np.dtype

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
    type: np.dtype


@dataclass(frozen=True)
class SingleDimRef(Expression):
    base: ValueRef
    dim: Constant

    def __post_init__(self):
        if not isinstance(self.dim, Constant) or not isinstance(self.dim.value, numbers.Integral):
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
        if not isinstance(left, ValueRef):
            msg = f"{left} is not a value reference."
            raise TypeError(msg)
        if not isinstance(right, ValueRef):
            msg = f"{right} is not a value reference"
            raise TypeError(msg)
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)

    @property
    def subexprs(self):
        yield self.left
        yield self.right


@dataclass(frozen=True)
class MinReduction(Expression):
    values: frozenset

    def __init__(self, *values):
        object.__setattr__(self, "values", frozenset(values))
        assert len(self.values) > 0

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
    values: frozenset

    def __init__(self, *values):
        object.__setattr__(self, "values", frozenset(values))
        assert len(self.values) > 0

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
    in_place: bool
    op: typing.ClassVar[str] = "NOT_IMPLEMENTED"

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
    in_place: typing.Optional[bool] = False
    op: typing.ClassVar[str] = "+"


@dataclass(frozen=True)
class SUB(BinOp):
    left: ValueRef
    right: ValueRef
    in_place: typing.Optional[bool] = False
    op: typing.ClassVar[str] = "-"


@dataclass(frozen=True)
class MULT(BinOp):
    left: ValueRef
    right: ValueRef
    in_place: typing.Optional[bool] = False
    op: typing.ClassVar[str] = "*"


@dataclass(frozen=True)
class TRUEDIV(BinOp):
    left: ValueRef
    right: ValueRef
    in_place: typing.Optional[bool] = False
    op: typing.ClassVar[str] = "/"


@dataclass(frozen=True)
class FLOORDIV(BinOp):
    left: ValueRef
    right: ValueRef
    in_place: typing.Optional[bool] = False
    op: typing.ClassVar[str] = "//"


@dataclass(frozen=True)
class MOD(BinOp):
    left: ValueRef
    right: ValueRef
    in_place: typing.Optional[bool] = False
    op: typing.ClassVar[str] = "%"


@dataclass(frozen=True)
class POW(BinOp):
    left: ValueRef
    right: ValueRef
    in_place: typing.Optional[bool] = False
    op: typing.ClassVar[str] = "**"


@dataclass(frozen=True)
class LSHIFT(BinOp):
    left: ValueRef
    right: ValueRef
    in_place: typing.Optional[bool] = False
    op: typing.ClassVar[str] = "<<"


@dataclass(frozen=True)
class RSHIFT(BinOp):
    left: ValueRef
    right: ValueRef
    in_place: typing.Optional[bool] = False
    op: typing.ClassVar[str] = ">>"


@dataclass(frozen=True)
class BITOR(BinOp):
    left: ValueRef
    right: ValueRef
    in_place: typing.Optional[bool] = False
    op: typing.ClassVar[str] = "|"


@dataclass(frozen=True)
class BITXOR(BinOp):
    left: ValueRef
    right: ValueRef
    in_place: typing.Optional[bool] = False
    op: typing.ClassVar[str] = "^"


@dataclass(frozen=True)
class BITAND(BinOp):
    left: ValueRef
    right: ValueRef
    in_place: typing.Optional[bool] = False
    op: typing.ClassVar[str] = "&"


@dataclass(frozen=True)
class MATMULT(BinOp):
    left: ValueRef
    right: ValueRef
    in_place: typing.Optional[bool] = False
    op: typing.ClassVar[str] = "@"


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
    op: typing.ClassVar[str] = "NOT_IMPLEMENTED"

    @property
    def subexprs(self):
        yield self.left
        yield self.right


@dataclass(frozen=True)
class EQ(CompareOp):
    left: ValueRef
    right: ValueRef
    op: typing.ClassVar[str] = "=="


@dataclass(frozen=True)
class NE(CompareOp):
    left: ValueRef
    right: ValueRef
    op: typing.ClassVar[str] = "!="


@dataclass(frozen=True)
class LT(CompareOp):
    left: ValueRef
    right: ValueRef
    op: typing.ClassVar[str] = "<"


@dataclass(frozen=True)
class LE(CompareOp):
    left: ValueRef
    right: ValueRef
    op: typing.ClassVar[str] = "<="


@dataclass(frozen=True)
class GT(CompareOp):
    left: ValueRef
    right: ValueRef
    op: typing.ClassVar[str] = ">"


@dataclass(frozen=True)
class GE(CompareOp):
    left: ValueRef
    right: ValueRef
    op: typing.ClassVar[str] = ">="


# Todo: These two are not correct. Fix..

@dataclass(frozen=True)
class IN(CompareOp):
    operand: ValueRef
    target: ValueRef
    op: typing.ClassVar[str] = "in"


@dataclass(frozen=True)
class NOTIN(CompareOp):
    operand: ValueRef
    target: ValueRef
    op: typing.ClassVar[str] = "not in"


class BoolOp(Expression):
    """
    Boolean operation using a single logical operation and an arbitrary
    number of operands. Base class is used here to aggregate type checks.

    """
    operands: typing.Iterable[ValueRef, ...]
    op: typing.ClassVar[str] = "NOT_IMPLEMENTED"
    c_op: typing.ClassVar[str] = "NOT_IMPLEMENTED"

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
    op: typing.ClassVar[str] = "or"

    def __init__(self, *operands):
        object.__setattr__(self, 'operands', frozenset(operands))


@dataclass(frozen=True)
class AND(BoolOp):
    """
    Boolean AND
    """
    operands: typing.Tuple[ValueRef, ...]
    op: typing.ClassVar[str] = "and"

    def __init__(self, *operands):
        object.__setattr__(self, 'operands', operands)


@dataclass(frozen=True)
class XOR(BoolOp):
    """
    Boolean XOR
    """
    operands: typing.Tuple[ValueRef, ...]
    op: typing.ClassVar[str] = "NOT_IMPLEMENTED" # there is no direct python equivalent

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
    op: typing.ClassVar[str] = ""

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
    op: typing.ClassVar[str] = "not"

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
        if not isinstance(self.start, ValueRef):
            msg = f"Start param of affine sequence must be a value reference. Received: {self.start}."
            raise ValueError(msg)
        if self.stop is not None and not isinstance(self.stop, ir.ValueRef):
            msg = f"Stop param of affine sequence must be a value reference. Received: {self.stop}."
            raise ValueError(msg)
        if not isinstance(self.step, ir.ValueRef):
            msg = f"Step param of affine sequence must be a value reference. Received: {self.step}."
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
            start = wrap_constant(0)
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


@dataclass(frozen=True)
class InPlaceOp(StmtBase):
    # Todo: set target explicitly for multiply accum which accumulates to expr.right here
    expr: BinOp
    pos: Position

    def __post_init__(self):
        assert isinstance(self.expr, BinOp) and self.expr.in_place

    @property
    def target(self):
        return self.expr.left

    @property
    def value(self):
        return self.expr.right


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
        if isinstance(self.value, BinOp):
            assert not self.value.in_place


@dataclass
class SingleExpr(StmtBase):
    expr: ValueRef
    pos: Position

    def __post_init__(self):
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


compare_ops = {EQ: "==",
               NE: "!=",
               LT: "<",
               LE: "<=",
               GT: ">",
               GE: ">=",
               IN: "in",
               NOTIN: "not in"
               }

binop_ops = {ADD: "+",
             SUB: "-",
             MULT: "*",
             TRUEDIV: "/",
             FLOORDIV: "//",
             POW: "**",
             }

inplace_binops = {ADD: "+=",
                  SUB: "-=",
                  MULT: "*=",
                  TRUEDIV: "/=",
                  FLOORDIV: "//=",
                  POW: "**="
                  }

unary_ops = {
    USUB: "-",
    UNOT: "~"
}

# constructable

constr_by_dtype = {
    np.dtype(np.int32): np.int32,
    np.dtype(np.int64): np.int64,
    np.dtype(np.float32): np.float32,
    np.dtype(np.float64): np.float64,
    np.dtype(np.bool_): np.bool_
}

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
