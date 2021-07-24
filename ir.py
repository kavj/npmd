from __future__ import annotations

import numbers
import operator
import typing
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property, singledispatchmethod


binaryops = frozenset({"+", "-", "*", "/", "//", "%", "**", "<<", ">>", "|", "^", "&", "@"})
inplace_ops = frozenset({"+=", "-=", "*=", "/=", "//=", "%=", "**=", "<<=", ">>=", "|=", "^=", "&=", "@="})
unaryops = frozenset({"+", "-", "~", "not"})
boolops = frozenset({"and", "or"})
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
class IntType:
    bitwidth: int
    is_array: typing.ClassVar[bool] = False
    is_integral: typing.ClassVar[bool] = True


@dataclass(frozen=True)
class FloatType:
    bitwidth: int
    is_array: typing.ClassVar[bool] = False
    is_integral: typing.ClassVar[bool] = False


# unfortunately this has to be tracked for lowering to C, where
# intrinsics must sometimes explicitly cast a predicate.

@dataclass(frozen=True)
class PredicateType:
    bitwidth: int
    is_array: typing.ClassVar[bool] = False
    # Treat predicates as integral by default, since they may be subjected to bit manipulation.
    is_integral: typing.ClassVar[bool] = True


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

    @property
    @abstractmethod
    def subexprs(self):
        raise NotImplementedError

    constant: clscond = False


class Constant(ValueRef):
    """
    Base class for anything that can be the target of an assignment. 

    """
    value: clscond = None
    constant: clscond = True

    def __bool__(self):
        return operator.truth(self.value)

    @property
    def subexprs(self):
        return
        yield


@dataclass(frozen=True)
class AttributeRef:
    """
    Limited number of attributes are read only.

    """
    value: NameRef
    attr: str
    constant: clscond = False


@dataclass(frozen=True)
class BoolNode(Constant):
    value: bool

    @property
    def subexprs(self):
        return
        yield


@dataclass(frozen=True)
class FloatNode(Constant):
    value: numbers.Real

    @property
    def subexprs(self):
        return
        yield


@dataclass(frozen=True)
class IntNode(Constant):
    value: numbers.Integral

    @property
    def subexprs(self):
        return
        yield


# commonly used

Zero = IntNode(0)
One = IntNode(1)

# Top Level


@dataclass(frozen=True)
class NameRef(ValueRef):
    # variable name ref
    name: str
    constant: clscond = False

    # empty generator to avoid checking
    # if something is an expression
    @property
    def subexprs(self):
        return
        yield


@dataclass(frozen=True)
class ArrayType:
    """
    descriptor for array inputs,
    This is necessary in order to describe array parameters on input without ambiguity.
    """

    ndims: IntNode
    dtype: typing.Union[IntType, FloatType, PredicateType]

    @property
    def array_type(self):
        return self

    def __post_init__(self):
        # Zero dimensions must be treated as a scalar, not an array.
        assert operator.gt(self.ndims.value, 0)


@dataclass(frozen=True)
class ArrayArg:
    """
    Array type descriptor

    dtype: scalar type used by this array
    dims: tuple(str or int,...)
    stride: input stride

    """
    dims: typing.Tuple[typing.Union[str, int, NameRef, IntNode], ...]
    dtype: typing.Union[type, IntType, FloatType, PredicateType]

    @property
    def array_type(self):
        return self

    @property
    def ndims(self):
        return len(self.dims)

    def __post_init__(self):
        # Zero dimensions must be treated as a scalar, not an array.
        assert len(self.dims) > 0


@dataclass(frozen=True)
class ArrayRef(ValueRef):
    name: NameRef
    array_type: ArrayType
    constant: clscond = False

    @property
    def base(self):
        return self

    @property
    def ndims(self):
        return self.array_type.ndims

    @property
    def subexprs(self):
        yield from ()


@dataclass(frozen=True)
class ViewRef:
    derived_from: typing.Union[ArrayRef, ViewRef]
    slice: typing.Optional[typing.Union[IntNode, Slice, NameRef, BinOp, UnaryOp]]
    array_type: ArrayType
    name: NameRef
    transposed: bool = False
    constant: clscond = False

    @cached_property
    def base(self):
        d = self.derived_from
        seen = {self}
        while isinstance(d, ViewRef):
            if d in seen:
                raise ValueError("contains view cycles")
            seen.add(d)
            d = d.derived_from
        return d

    @cached_property
    def dtype(self):
        return self.base.dtype

    @cached_property
    def ndims(self):
        return self.array_type.array_type.ndims

    @cached_property
    def name(self):
        return self.base.name

    @property
    def subexprs(self):
        return
        yield


@dataclass(frozen=True)
class Length(ValueRef):
    value: ValueRef
    constant: clscond = False

    @property
    def subexprs(self):
        yield self.value


@dataclass(frozen=True)
class Subscript(ValueRef):
    value: ValueRef
    slice: ValueRef
    constant: clscond = False

    @property
    def subexprs(self):
        yield self.value
        yield self.slice


@dataclass(frozen=True)
class Min(ValueRef):
    """
    Min iteration count over a number of counters
    """
    values: typing.Tuple[ValueRef, ...]

    @property
    def subexprs(self):
        for subexpr in self.values:
            yield subexpr


@dataclass(frozen=True)
class Max(ValueRef):
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

    funcs: typing.List[Function]
    imports: typing.List[typing.Any]


@dataclass(frozen=True)
class Slice(ValueRef):
    """
    IR representation of a slice.

    Per dimension constraints can be applied

    """

    start: typing.Union[NameRef, IntNode]
    stop: typing.Optional[typing.Union[NameRef, IntNode]]
    step: typing.Union[NameRef, IntNode]

    @property
    def subexprs(self):
        yield self.start
        yield self.stop
        yield self.step


@dataclass(frozen=True)
class Tuple(ValueRef):
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
class BinOp(ValueRef):
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


@dataclass(frozen=True)
class BoolOp(ValueRef):
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
class Call(ValueRef):
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

    @property
    def arg_count(self):
        return len(self.args)

    @property
    def kwarg_count(self):
        return len(self.keywords)

    def has_keyword(self, kw):
        return any(keyword == kw for (keyword, _) in self.keywords)


@dataclass(frozen=True)
class AffineSeq(ValueRef):
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
class Ternary(ValueRef):
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
class Reversed(ValueRef):
    """
    Sentinel for a "reversed" object

    """

    iterable: ValueRef

    @property
    def subexprs(self):
        yield self.iterable


@dataclass(frozen=True)
class UnaryOp(ValueRef):
    operand: ValueRef
    op: str

    def __post_init__(self):
        assert self.op in unaryops

    @property
    def subexprs(self):
        yield self.operand


@dataclass(frozen=True)
class Zip(ValueRef):
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
    target: typing.Any
    iterable: AffineSeq
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
    mod: str
    asname: str
    pos: Position


@dataclass
class NameImport(StmtBase):
    mod: str
    name: str
    asname: str
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


# utility nodes

@dataclass(frozen=True)
class Max(ValueRef):
    exprs: typing.Tuple[typing.Union[NameRef, ValueRef], ...]

    def subexprs(self):
        for subexpr in self.exprs:
            yield subexpr


@dataclass(frozen=True)
class Min(ValueRef):
    exprs: typing.Tuple[typing.Union[NameRef, ValueRef], ...]

    @property
    def subexprs(self):
        if isinstance(self.exprs, Iterable):
            for subexpr in self.exprs:
                yield subexpr
        else:
            yield self.exprs
