from __future__ import annotations

import math
import numbers
import operator
import typing
from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum, auto, unique
from functools import cached_property

import ir
from errors import CompilerError

# Note: operations are split this way to avoid incorrectly accepting bad inplace to out of place conversions.
binary_ops = frozenset({"+", "-", "*", "/", "//", "%", "**", "<<", ">>", "|", "^", "&", "@"})
bool_ops = frozenset({"and", "or", "xor"})
compare_ops = frozenset({"==", "!=", "<", "<=", ">", ">=", "is", "isnot"})
in_place_ops = frozenset({"+=", "-=", "*=", "/=", "//=", "%=", "**=", "<<=", ">>=", "|=", "^=", "&=", "@="})
unary_ops = frozenset({"-", "~"})

# sub-taxonomies
add_ops = frozenset({"+", "+="})
bit_shift_ops = frozenset({"<<", ">>", "<<=", ">>="})
divide_ops = frozenset({"/", "//", "/=", "//="})
multiply_ops = frozenset({"*", "*="})
modulo_ops = frozenset({"%", "%="})
subtract_ops = frozenset({"-", "-="})
floor_divide_ops = frozenset({"//", "//="})
shift_left_ops = frozenset({"<<", "<<="})
shift_right_ops = frozenset({">>", ">>="})
true_divide_ops = frozenset({"/", "/="})

# conversions
out_of_place_to_in_place = {
    "+": "+=",
    "-": "-=",
    "*": "*=",
    "/": "/=",
    "//": "//=",
    "%": "%=",
    "**": "**=",
    "<<": "<<=",
    ">>": ">>=",
    "|": "|=",
    "^": "^=",
    "&": "&=",
    "@": "@=",
}

in_place_to_out_of_place = {
    "+=": "+",
    "-=": "-",
    "*=": "*",
    "/=": "/",
    "//=": "//",
    "%=": "%",
    "**=": "**",
    "<<=": "<<",
    ">>=": ">>",
    "|=": "|",
    "^=": "^",
    "&=": "&",
    "@=": "@",
}

supported_builtins = frozenset({'iter', 'range', 'enumerate', 'zip', 'all', 'any', 'max', 'min', 'abs', 'pow',
                                'round', 'reversed'})


@unique
class Sentinels(Enum):
    UNSUPPORTED = auto()
    NONE = auto()


@dataclass(frozen=True)
class Position:
    line_begin: int
    line_end: int
    col_begin: int
    col_end: int


clscond = typing.ClassVar[bool]


@dataclass(frozen=True)
class ScalarType:
    bits: int
    integral: bool
    boolean: bool


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

    def __bool__(self):
        return operator.truth(self.value)


@dataclass(frozen=True)
class BoolConst(Constant):
    value: bool

    def __post_init__(self):
        assert isinstance(self.value, bool)


@dataclass(frozen=True)
class FloatConst(Constant):
    value: numbers.Real

    def __post_init__(self):
        assert isinstance(self.value, numbers.Real)


@dataclass(frozen=True)
class IntConst(Constant):
    value: numbers.Integral

    def __post_init__(self):
        assert isinstance(self.value, numbers.Integral)


@dataclass(frozen=True)
class StringConst(Constant):
    value: str

    def __post_init__(self):
        assert isinstance(self.value, str)
        if any(ord(v) > 127 for v in self.value):
            msg = f"Only strings that can be converted to ascii text are supported. This is mainly intended" \
                  f"to facilitate simple printing support at some point."
            raise CompilerError(msg)


# commonly used

Zero = IntConst(0)
One = IntConst(1)
Neg_One = IntConst(-1)
NAN = FloatConst(math.nan)
TRUE = BoolConst(True)
FALSE = BoolConst(False)


# Top Level

# Todo: add typing to these to avoid needing constant access to symbol table
@dataclass(frozen=True)
class NameRef(ValueRef):
    # variable name ref
    name: str

    def __post_init__(self):
        assert isinstance(self.name, str)


@dataclass(frozen=True)
class ArrayType(ValueRef):
    ndims: int
    dtype: typing.Hashable

    def __post_init__(self):
        assert isinstance(self.ndims, int)
        assert self.ndims > 0
        assert isinstance(self.dtype, Hashable)


@dataclass(frozen=True)
class ArrayInit(Expression):
    ndims: int
    dtype: typing.Hashable
    dims: typing.Tuple[typing.Union[NameRef, IntConst], ...]
    fill_value: typing.Optional[typing.Union[IntConst, FloatConst, BoolConst]]

    def __post_init__(self):
        assert isinstance(self.ndims, numbers.Integral)
        assert self.dims is None or isinstance(self.dims, tuple)
        assert isinstance(self.dtype, Hashable)

    @property
    def subexprs(self):
        for d in self.dims:
            yield d


@dataclass(frozen=True)
class ArrayArg(Expression):
    ndims: int
    dtype: typing.Hashable
    dims: typing.Optional[typing.Tuple[typing.Tuple[int, int], ...]]
    evol: typing.Optional[str]

    @property
    def subexprs(self):
        for d in self.dims:
            yield d


@dataclass(frozen=True)
class ArrayRef(Expression):
    name: NameRef
    dims: typing.Tuple[typing.Union[NameRef, IntConst], ...]
    dtype: typing.Hashable

    def __post_init__(self):
        assert isinstance(self.name, NameRef)
        assert isinstance(self.dims, tuple)
        assert isinstance(self.dtype, Hashable)

    @property
    def base(self):
        return self

    @property
    def ndims(self):
        return len(self.dims)

    @property
    def subexprs(self):
        for d in self.dims:
            yield d


@dataclass(frozen=True)
class SingleDimRef(Expression):
    base: ValueRef
    dim: IntConst

    def __post_init__(self):
        if not isinstance(self.dim, IntConst):
            msg = f"Expected integer constant, received {self.dim} of type {type(self.dim)}."
            raise TypeError(msg)

    @property
    def subexprs(self):
        yield self.base
        yield self.dim


# Todo: changes here should be bested as the component yielded by loop unpacking.

@dataclass(frozen=True)
class IteratedViewRef:
    expr: ValueRef
    base: typing.Union[ArrayRef, ViewRef]
    iterated: typing.Optional[bool] = True
    transpose: typing.Optional[bool] = False

    def __post_init__(self):
        assert isinstance(self.base, (ArrayRef, ViewRef))
        assert self.slice is None or isinstance(self.slice, (IntConst, Slice, NameRef, BinOp, UnaryOp))

    @cached_property
    def dtype(self):
        return self.base.dtype


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
    slice: ValueRef

    def __post_init__(self):
        if not isinstance(self.value, ValueRef):
            msg = f"Expected ValueRef, got {self.value}, type: {type(self.value)}."
            raise TypeError(msg)
        elif not isinstance(self.slice, ValueRef):
            msg = f"Expected ValueRef, got {self.slice}, type: {type(self.slice)}."
            raise TypeError(msg)

    @property
    def subexprs(self):
        yield self.value
        yield self.slice


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
    values: typing.Union[typing.Set, typing.FrozenSet]

    def __init__(self, *values):
        if len(values) == 1:
            values, = values
            if isinstance(values, set):
                values = frozenset(values)
            object.__setattr__(self, "values", values)
        else:
            object.__setattr__(self, "values", frozenset(v for v in values))

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
    values: typing.Union[typing.Set, typing.FrozenSet]

    def __init__(self, *values):
        if len(values) == 1:
            values, = values
            if isinstance(values, set):
                values = frozenset(values)
            object.__setattr__(self, "values", values)
        else:
            object.__setattr__(self, "values", frozenset(v for v in values))

    @property
    def subexprs(self):
        for v in self._values:
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

    start: typing.Union[NameRef, IntConst]
    stop: typing.Optional[typing.Union[NameRef, IntConst]]
    step: typing.Union[NameRef, IntConst]

    def __post_init__(self):
        assert isinstance(self.start, (NameRef, IntConst))
        assert isinstance(self.stop, (NameRef, IntConst))
        assert isinstance(self.step, (NameRef, IntConst))

    @property
    def subexprs(self):
        yield self.start
        yield self.stop
        yield self.step


@dataclass(frozen=True)
class Tuple(Expression):
    elements: typing.Tuple[ValueRef, ...]

    def __post_init__(self):
        assert isinstance(self.elements, tuple)
        assert all(isinstance(e, ValueRef) for e in self.elements)

    @property
    def subexprs(self):
        for e in self.elements:
            yield e

    @property
    def length(self):
        return len(self.elements)


Targetable = typing.TypeVar('Targetable', NameRef, Subscript, Tuple)


@dataclass(frozen=True)
class BinOp(Expression):
    left: ValueRef
    right: ValueRef
    op: str

    def __post_init__(self):
        assert (self.op in binary_ops or self.op in in_place_ops)
        assert isinstance(self.right, ValueRef)
        if self.in_place:
            # ensure things like -a += b are treated as compiler errors.
            # These shouldn't even pass the parser if caught in source but
            # could be accidentally created internally.
            assert isinstance(self.left, (NameRef, Subscript))
        else:
            assert isinstance(self.left, ValueRef)

    @property
    def subexprs(self):
        yield self.left
        yield self.right

    @cached_property
    def in_place(self):
        return self.op in in_place_ops


# Compare ops are once again their own class,
# since they cannot be in place like binops


@dataclass(frozen=True)
class CompareOp(Expression):
    left: ValueRef
    right: ValueRef
    op: str

    def __post_init__(self):
        assert self.op in compare_ops
        assert isinstance(self.left, ValueRef)
        assert isinstance(self.right, ValueRef)

    @property
    def subexprs(self):
        yield self.left
        yield self.right


class BoolOp(Expression, ABC):
    """
    Boolean operation using a single logical operation and an arbitrary
    number of operands. Base class is used here to aggregate type checks.

    """

    pass


@dataclass(frozen=True)
class OR(BoolOp):
    """
    Boolean OR
    """
    operands: typing.Tuple[ValueRef, ...]

    def __post_init__(self):
        assert (isinstance(self.operands, tuple))
        assert len(self.operands) >= 2
        assert all(isinstance(operand, ValueRef) for operand in self.operands)

    @property
    def subexprs(self):
        for operand in self.operands:
            yield operand


@dataclass(frozen=True)
class AND(BoolOp):
    """
    Boolean AND
    """
    operands: typing.Tuple[ValueRef, ...]

    def __post_init__(self):
        assert (isinstance(self.operands, tuple))
        assert len(self.operands) >= 2
        assert all(isinstance(operand, ValueRef) for operand in self.operands)

    @property
    def subexprs(self):
        for operand in self.operands:
            yield operand


@dataclass(frozen=True)
class XOR(BoolOp):
    """
    Boolean XOR
    """
    operands: typing.Tuple[ValueRef, ...]

    def __post_init__(self):
        assert (isinstance(self.operands, tuple))
        assert len(self.operands) >= 2
        assert all(isinstance(operand, ValueRef) for operand in self.operands)

    @property
    def subexprs(self):
        for operand in self.operands:
            yield operand


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

    func: NameRef
    args: typing.Tuple[ValueRef, ...]
    # expressions must be safely hashable, so we can't use a dictionary here
    keywords: typing.Tuple[typing.Tuple[str, ValueRef], ...]

    def __post_init__(self):
        assert isinstance(self.args, tuple)
        assert isinstance(self.keywords, tuple)
        assert all(isinstance(arg, ValueRef) for arg in self.args)
        assert all(isinstance(arg, ValueRef) for (key, arg) in self.keywords)

    @property
    def subexprs(self):
        for arg in self.args:
            yield arg
        for kw, arg in self.keywords:
            yield arg

    def has_keyword(self, kw):
        return any(keyword == kw for (keyword, _) in self.keywords)


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


@dataclass(frozen=True)
class UnaryOp(Expression):
    operand: ValueRef
    op: str

    def __post_init__(self):
        assert self.op in unary_ops
        assert isinstance(self.operand, ValueRef)

    @property
    def subexprs(self):
        yield self.operand


@dataclass(frozen=True)
class Enumerate(Expression):
    """
    High level sentinel, representing an enumerate object.
    """

    iterable: ValueRef
    start: ValueRef

    def __post_init__(self):
        assert isinstance(self.iterable, ValueRef)
        assert isinstance(self.start, ValueRef)

    @property
    def subexprs(self):
        yield self.iterable
        yield self.start


@dataclass(frozen=True)
class Zip(Expression):
    """
    High level sentinel representing a zip object.
    """

    elements: typing.Tuple[ValueRef, ...]

    def __post_init__(self):
        assert isinstance(self.elements, tuple)
        assert all(isinstance(e, ValueRef) for e in self.elements)

    @property
    def subexprs(self):
        for e in self.elements:
            yield e

    @property
    def length(self):
        return len(self.elements)


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

    @property
    def in_place(self):
        return isinstance(self.value, BinOp) and self.value.in_place


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


@dataclass(frozen=True)
class ParallelLoop(StmtBase):
    loop: ForLoop

    def __post_init__(self):
        if not isinstance(loop, ForLoop):
            raise CompilerError("Can only parallelize for loops.")


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
    count: IntConst

    def subexprs(self):
        yield self.array
        yield self.stride


@dataclass(frozen=True)
class unroll(ir.Expression):
    array: ValueRef
    count: IntConst

    def subexprs(self):
        yield self.array


@dataclass(frozen=True)
class interleave(ir.Expression):
    arrays: typing.Tuple[ir.ValueRef, ...]

    def subexprs(self):
        for a in self.arrays:
            yield a
