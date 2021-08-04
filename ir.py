from __future__ import annotations

import numbers
import operator
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property


binaryops = frozenset({"+", "-", "*", "/", "//", "%", "**", "<<", ">>", "|", "^", "&", "@"})
inplace_ops = frozenset({"+=", "-=", "*=", "/=", "//=", "%=", "**=", "<<=", ">>=", "|=", "^=", "&=", "@="})
unaryops = frozenset({"+", "-", "~", "not"})
boolops = frozenset({"and", "or", "xor"})
compareops = frozenset({"==", "!=", "<", "<=", ">", ">=", "is", "isnot", "in", "notin"})

oop_to_inplace = {
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

inplace_to_oop = {
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


# commonly used

Zero = IntConst(0)
One = IntConst(1)


# Top Level


@dataclass(frozen=True)
class NameRef(ValueRef):
    # variable name ref
    name: str


@dataclass(frozen=True)
class ArrayType(ValueRef):
    ndims: int
    dtype: typing.Hashable


@dataclass(frozen=True)
class ArrayRef(ValueRef):
    name: NameRef
    dims: typing.Tuple[typing.Union[NameRef, IntConst], ...]
    dtype: typing.Hashable

    @property
    def base(self):
        return self

    @property
    def ndims(self):
        return len(self.dims)


@dataclass(frozen=True)
class ViewRef:
    # name: NameRef
    base: typing.Union[ArrayRef, ViewRef]
    slice: typing.Optional[typing.Union[IntConst, Slice, NameRef, BinOp, UnaryOp]]
    transposed: bool

    @cached_property
    def dtype(self):
        return self.base.dtype

    @cached_property
    def ndims(self):
        if isinstance(self.slice, Slice):
            return self.base.ndims
        else:
            return self.base.ndims - 1

    @cached_property
    def name(self):
        return self.base.name


@dataclass(frozen=True)
class Length(Expression):
    value: ValueRef

    @property
    def subexprs(self):
        yield self.value


@dataclass(frozen=True)
class Subscript(Expression):
    value: ValueRef
    slice: ValueRef

    @property
    def subexprs(self):
        yield self.value
        yield self.slice


@dataclass(frozen=True)
class Min(Expression):
    """
    Min iteration count over a number of counters
    """
    values: typing.Tuple[ValueRef, ...]

    @property
    def subexprs(self):
        for subexpr in self.values:
            yield subexpr


@dataclass(frozen=True)
class Max(Expression):
    """
    Max iteration count over a number of counters
    """
    values: typing.Tuple[ValueRef, ...]

    @property
    def subexprs(self):
        for subexpr in self.values:
            yield subexpr


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

    functions: typing.List[Function]
    imports: typing.List[typing.Any]


@dataclass(frozen=True)
class Slice(Expression):
    """
    IR representation of a slice.

    Per dimension constraints can be applied

    """

    start: typing.Union[NameRef, IntConst]
    stop: typing.Optional[typing.Union[NameRef, IntConst]]
    step: typing.Union[NameRef, IntConst]

    @property
    def subexprs(self):
        yield self.start
        yield self.stop
        yield self.step


@dataclass(frozen=True)
class Tuple(Expression):
    elements: typing.Tuple[ValueRef, ...]

    def __post_init__(self):
        assert (isinstance(self.elements, tuple))

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
        assert (self.op in binaryops or self.op in inplace_ops or self.op in compareops)

    @property
    def subexprs(self):
        yield self.left
        yield self.right

    @cached_property
    def is_compare_op(self):
        return self.op in compareops

    @cached_property
    def in_place(self):
        return self.op in inplace_ops


# Compare ops are once again their own class,
# since they cannot be in place like binops


@dataclass(frozen=True)
class CompareOp(Expression):
    left: ValueRef
    right: ValueRef
    op: str

    def subexprs(self):
        yield self.left
        yield self.right


@dataclass(frozen=True)
class OR(Expression):
    """
    Boolean OR
    """
    operands: typing.Tuple[ValueRef, ...]

    def __post_init__(self):
        assert (isinstance(self.operands, tuple))
        assert len(self.operands) >= 2

    @property
    def subexprs(self):
        for operand in self.operands:
            yield operand


@dataclass(frozen=True)
class AND(Expression):
    """
    Boolean AND
    """
    operands: typing.Tuple[ValueRef, ...]

    def __post_init__(self):
        assert (isinstance(self.operands, tuple))
        assert len(self.operands) >= 2

    @property
    def subexprs(self):
        for operand in self.operands:
            yield operand


@dataclass(frozen=True)
class XOR(Expression):
    """
    Boolean XOR
    """
    operands: typing.Tuple[ValueRef, ...]

    def __post_init__(self):
        assert (isinstance(self.operands, tuple))
        assert len(self.operands) >= 2

    @property
    def subexprs(self):
        for operand in self.operands:
            yield operand


@dataclass(frozen=True)
class TRUTH(Expression):
    """
    Truth test single operand
    """
    operand: ValueRef

    @property
    def subexprs(self):
        yield self.operand


@dataclass(frozen=True)
class NOT(Expression):
    """
    Boolean not
    """

    operand: ValueRef


@dataclass(frozen=True)
class BoolOp(Expression):
    """
    Boolean operation using a single logical operation and an arbitrary
    number of operands. 

    Comparisons between expressions are compared without modification.

    """

    operands: typing.Tuple[ValueRef, ...]
    op: str

    def __post_init__(self):
        assert (isinstance(self.operands, tuple))
        assert self.op in boolops

    @property
    def subexprs(self):
        for operand in self.operands:
            yield operand


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
        assert (isinstance(self.args, tuple))

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
class Ternary(Expression):
    """
    A Python if expression.
    
    """

    test: ValueRef
    if_expr: ValueRef
    else_expr: ValueRef

    @property
    def subexprs(self):
        yield self.if_expr
        yield self.test
        yield self.else_expr


@dataclass(frozen=True)
class Reversed(Expression):
    """
    Sentinel for a "reversed" object

    """

    iterable: ValueRef

    @property
    def subexprs(self):
        yield self.iterable


@dataclass(frozen=True)
class UnaryOp(Expression):
    operand: ValueRef
    op: str

    def __post_init__(self):
        assert self.op in unaryops

    @property
    def subexprs(self):
        yield self.operand


@dataclass(frozen=True)
class Zip(Expression):
    """
    High level sentinel representing a zip object. This is the only unpackable iterator type in this IR.
    Enumerate(object) is handled by Zip(AffineSeq, object).
    """

    elements: typing.Tuple[ValueRef, ...]

    def __post_init__(self):
        assert (isinstance(self.elements, tuple))

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

    @property
    def in_place(self):
        return isinstance(self.value, BinOp) and self.value.in_place


@dataclass
class SingleExpr(StmtBase):
    expr: ValueRef
    pos: Position


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


@dataclass
class IfElse(StmtBase):
    test: ValueRef
    if_branch: typing.List[Statement]
    else_branch: typing.List[Statement]
    pos: Position


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


@dataclass
class Pass(StmtBase):
    pos: Position


@dataclass
class Return(StmtBase):
    value: typing.Optional[ValueRef]
    pos: Position


@dataclass
class WhileLoop(StmtBase):
    test: ValueRef
    body: typing.List[Statement]
    pos: Position
