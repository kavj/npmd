from __future__ import annotations

import numpy as np
import numbers
import operator
import typing
from abc import ABC
from collections.abc import Iterable
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
class ImportRef:
    """
    module_name: Name of a module, which may be package.subpackage.module
    imported_name: Name to import from this module, if any. If nothing is imported, this means only the module
                 name enters scope.
    as_name: Alias used to refer to the module or imported name in place of the original name.
    """
    module_name: NameRef
    imported_name: typing.Optional[NameRef]
    as_name: typing.Optional[NameRef]

    @property
    def is_module(self):
        return self.imported_name is None

    @property
    def bound_name(self):
        return (self.as_name if self.as_name is not None
                else self.imported_name if self.imported_name is not None
                else self.module_name)


# Types are subject to change.
# Predicate types included for bookkeeping, as mask types used in SIMD ISAs
# are not always compatible with C99 bool.

class Int32:
    bits = 32
    min_value = -2**31
    max_value = 2**31-1

    def __hash__(self):
        return hash((self.__class__.__name__, self.bits))


class Int64:
    bits = 64
    min_value = -2**63
    max_value = 2**63-1

    def __hash__(self):
        return hash((self.__class__.__name__, self.bits))


class Float32:
    bits = 64
    min_value = np.finfo(np.float32).min
    max_value = np.finfo(np.float32).max

    def __hash__(self):
        return hash((self.__class__.__name__, self.bits))


class Float64:
    bits = 64
    min_value = np.finfo(np.float64).min
    max_value = np.finfo(np.float64).max

    def __hash__(self):
        return hash((self.__class__.__name__, self.bits))


class Predicate32:
    bits = 32

    def __hash__(self):
        return hash((self.__class__.__name__, 64))


class Predicate64:
    bits = 64

    def __hash__(self):
        return hash((self.__class__.__name__, 64))


class BoolType:
    bits = 8

    def __hash__(self):
        return hash((self.__class__.__name__, 8))


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


@dataclass(frozen=True)
class FloatConst(Constant):
    value: numbers.Real


@dataclass(frozen=True)
class IntConst(Constant):
    value: numbers.Integral


# commonly used

Zero = IntConst(0)
One = IntConst(1)


# Top Level


@dataclass(frozen=True)
class NameRef:
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
class Length(ValueRef):
    value: ValueRef

    @property
    def subexprs(self):
        yield self.value


@dataclass(frozen=True)
class Subscript(ValueRef):
    value: ValueRef
    slice: ValueRef

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

    start: typing.Union[NameRef, IntConst]
    stop: typing.Optional[typing.Union[NameRef, IntConst]]
    step: typing.Union[NameRef, IntConst]

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


# Compare ops are once again their own class,
# since they cannot be in place like binops


@dataclass(frozen=True)
class CompareOp(ValueRef):
    left: ValueRef
    right: ValueRef
    op: str

    def subexprs(self):
        yield self.left
        yield self.right


@dataclass(frozen=True)
class OR(ValueRef):
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
class AND(ValueRef):
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
class XOR(ValueRef):
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
class TRUTH(ValueRef):
    """
    Truth test single operand
    """
    operand: ValueRef

    @property
    def subexprs(self):
        yield self.operand


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
        for kw, arg in self.keywords:
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


@dataclass(frozen=True)
class InductionVar(ValueRef):
    # Note: These must already be linearized, because
    # this won't recursively unpack subexpressions.
    targets: ValueRef
    iterables: ValueRef

    @property
    def subexprs(self):
        if hasattr(self.targets, 'subexprs') and hasattr(self.iterables, 'subexprs'):
            for target, iterable in zip(self.targets.subexprs, self.iterables.subexprs):
                yield target, iterable
        else:
            yield self.targets, self.iterables


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
