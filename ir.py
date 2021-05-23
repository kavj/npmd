from __future__ import annotations

import itertools
import numbers
import operator
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

binaryops = frozenset({"+", "-", "*", "/", "//", "%", "**", "<<", ">>", "|", "^", "&", "@"})
inplace_ops = frozenset({"+=", "-=", "*=", "/=", "//=", "%=", "**=", "<<=", ">>=", "|=", "^=", "&=", "@="})
unaryops = frozenset({"+", "-", "~", "not"})
boolops = frozenset({"and", "or"})
compareops = frozenset({"==", "!=", "<", "<=", ">", ">=", "is", "isnot", "in", "notin"})

supported_builtins = frozenset({'iter', 'range', 'enumerate', 'zip', 'all', 'any', 'max', 'min', 'abs', 'pow',
                                'round', 'reversed'})


@dataclass(frozen=True)
class Position:
    line_begin: int
    line_end: int
    col_begin: int
    col_end: int


clscond = typing.ClassVar[bool]


class Walkable(ABC):

    @abstractmethod
    def walk(self):
        raise NotImplementedError


class StmtBase:
    is_loop_entry: clscond = False  # ForLoop, WhileLoop
    is_terminator: clscond = False  # Continue, Break, Return
    # statments that overwrite variable names or array indices within scope
    clobbers: clscond = False  # Assign, ForLoop


Statement = typing.TypeVar('Statement', bound=StmtBase)


class Expression:
    """
    This is the start of a base class for expressions.
    
    For our purposes, we distinguish an expression from a value by asking whether it can be 
    an assignment target. 

    The term 'expression_like' is used elsewhere in the code to include mutable targets, where
    assignment alters a data structure as opposed to simply binding a value to a name.

    """

    subscripted: clscond = False

    @property
    @abstractmethod
    def subexprs(self):
        raise NotImplementedError

    @property
    def constant(self):
        return all(se.constant for se in self.subexprs)

    def post_order_walk(self):
        """
        Walk sub-expressions of node in post order, ignoring duplicates.

        """

        seen = set()
        queued: typing.List[typing.Tuple[typing.Optional[Expression], typing.Generator]] = [(None, self.subexprs)]

        while queued:
            try:
                e = next(queued[-1][1])
                if e in seen:
                    continue
                seen.add(e)
                if isinstance(e, Expression):
                    queued.append((e, e.subexprs))
                else:
                    yield e
            except StopIteration:
                e, _ = queued.pop()
                # ignore the original node we're walking
                if queued:
                    yield e


class Constant:
    """
    Base class for anything that can be the target of an assignment. 

    """
    constant: clscond = True
    subscripted: clscond = False
    value: typing.ClassVar = None

    def __bool__(self):
        return operator.truth(self.value)


@dataclass(frozen=True)
class AttributeRef:
    """
    Limited number of attributes are read only.

    """
    value: NameRef
    attr: typing.Tuple[str]
    constant: clscond = False


@dataclass(frozen=True)
class BoolNode(Constant):
    value: bool
    constant: clscond = True


@dataclass(frozen=True)
class EllipsisNode(Constant):
    # singleton class
    value: typing.Any = Ellipsis


@dataclass(frozen=True)
class FloatNode(Constant):
    value: numbers.Real


@dataclass(frozen=True)
class IntNode(Constant):
    value: numbers.Integral


@dataclass(frozen=True)
class StringNode(Constant):
    value: str


# Top Level

@dataclass(frozen=True)
class NameRef:
    # variable name ref
    name: typing.Union[str, AttributeRef]
    constant: clscond = False
    subscripted: clscond = False


@dataclass(frozen=True)
class ArrayRef:
    # This is necessary to easily describe the type of an otherwise anonymous subscript,
    # particularly in cases of subscripts that are not bound to explicit array references.
    name: NameRef
    dtype: type
    ndims: int
    dims: typing.Optional[typing.Tuple[IntNode, ...]]
    constant: clscond = False
    subscripted: clscond = False

    def __post_init__(self):
        assert (self.dims is None or len(self.dims) == self.ndims)

    @property
    def base(self):
        return self


@dataclass(frozen=True)
class ViewRef:
    derived_from: typing.Union[ArrayRef, ViewRef]
    subscript: Subscript
    transposed: bool = False

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

    # these could be cached_property

    @cached_property
    def dtype(self):
        return self.base.dtype

    @cached_property
    def ndims(self):
        d = self.derived_from
        seen = {self}
        reduce_by = 1 if isinstance(self.subscript, (NameRef, IntNode)) else 0
        while isinstance(d, ViewRef):
            if d in seen:
                raise ValueError("contains view cycles")
            # this may need work
            if isinstance(d.subscript, (NameRef, IntNode)):
                reduce_by += 1
            seen.add(d)
            d = d.derived_from
        # negative if the array is over-subscripted
        return d.ndims - reduce_by

    @cached_property
    def name(self):
        return self.base.name


ValueRef = typing.TypeVar('ValueRef', Expression, Constant, NameRef, AttributeRef, ArrayRef, ViewRef)


@dataclass(frozen=True)
class Subscript(Expression):
    value: ValueRef
    slice: ValueRef
    constant: clscond = False
    subscripted: clscond = True

    @property
    def subexprs(self):
        yield self.value
        yield self.slice


@dataclass(frozen=True)
class Argument:
    name: NameRef
    annot: str = None
    commt: str = None
    defaultvalue: typing.Any = None
    constant: clscond = False
    subscripted: clscond = False


@dataclass(frozen=True)
class ShapeRef(Expression):
    array: typing.Any
    dim: typing.Optional[ValueRef] = None

    @property
    def ndims(self):
        return self.array.ndims

    @property
    def subexprs(self):
        yield self.array
        yield self.dim

    # not quite constant since it can refer to single definitions
    # which appear in loops
    constant: typing.ClassVar[bool] = False


@dataclass
class Function(Walkable):
    name: str
    args: typing.List[Argument]
    body: typing.List[Statement]

    def walk(self):
        for n in self.body:
            yield n


@dataclass
class Module:
    """
    This aggregates imports and Function graphs

    """

    funcs: typing.List[Function]
    imports: typing.List[typing.Any]


@dataclass(frozen=True)
class Slice(Expression):
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
class Unsupported(Expression):
    name: str
    msg: str

    # dummy class, just treats expression as a protocol
    @property
    def subexprs(self):
        raise StopIteration


@dataclass(frozen=True)
class Tuple(Expression):
    """
    High level sentinel matching a tuple.
    These cannot be directly assigned but may be unpacked as they enter scope.

    They also arise in loop unpacking and variable permutations.

    For example, constructs such as zip generate tuple valued outputs, which are typically immediately unpacked.

    for u,v in zip(a,b):
       ...

    Tuples can also be used to indicate variable permutations.

    b, a, d, c = a, b, c, d

    Tuples are commonly used to hold array dimensions.

    dim_zero, dim_one = array.shape

    """

    elements: typing.Tuple[ValueRef, ...]

    def __post_init__(self):
        assert (isinstance(self.elements, tuple))

    @property
    def subexprs(self):
        for e in self.elements:
            yield e


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

    funcname: str
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
class MinConstraint(Expression):
    """
    used primarily for lowering for loops with zip and enumerate
    """
    constraints: set

    @property
    def subexprs(self):
        for c in self.constraints:
            yield c


@dataclass(frozen=True)
class Counter(Expression):
    """
    This is used to distinguish either an unpacked counter of an enumerate
    expression or a range operation. For our purposes, the difference is that
    range may have a non-unit step and must have a stop parameter.


    This is used to distinguish an unpacked counter of an enumerate expression.
    It's primarily used to assist in unpacking. Otherwise the common pattern

    for index, value in enumerate(iterable):
        ...

    has the serialized header representation

    [(index, Counter(IntNode(0), None, IntNode(1))), (value, iterable)]

    """

    start: ValueRef
    stop: typing.Optional[ValueRef]
    step: ValueRef

    is_iterator_like: clscond = True
    is_counter: clscond = True

    @property
    def subexprs(self):
        yield self.start
        if self.stop is not None:
            yield self.stop
        yield self.step


@dataclass(frozen=True)
class IfExpr(Expression):
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
    is_iterator_like: clscond = True

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
    Enumerate(object) is handled by Zip(Counter, object).
    """

    elements: typing.Tuple[ValueRef, ...]

    def __post_init__(self):
        assert (isinstance(self.elements, tuple))

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
    # Record optional annotation, since we need to check
    # reachability before applying.
    annot: str = None
    clobbers: clscond = True

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
    is_terminator: clscond = True


@dataclass
class Continue(StmtBase):
    pos: Position
    is_terminator: clscond = True


@dataclass
class ForLoop(StmtBase, Walkable):
    assigns: typing.List[typing.Tuple[Targetable, ValueRef]]
    body: typing.List[Statement]
    pos: Position
    is_loop_entry: clscond = True
    clobbers: clscond = True

    def walk(self):
        for stmt in self.body:
            yield stmt

    def walk_assignments(self):
        for target, value in self.assigns:
            yield target, value


@dataclass
class IfElse(StmtBase, Walkable):
    test: ValueRef
    if_branch: typing.List[Statement]
    else_branch: typing.List[Statement]
    pos: Position
    is_conditional_branch: clscond = True

    def walk(self):
        for stmt in itertools.chain(self.if_branch, self.else_branch):
            yield stmt


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
    is_terminator: clscond = True


@dataclass
class WhileLoop(StmtBase, Walkable):
    test: ValueRef
    body: typing.List[Statement]
    pos: Position
    is_loop_entry: clscond = True

    def walk(self):
        for n in self.body:
            yield n


# utility nodes

@dataclass(frozen=True)
class IntegralType:
    bit_width: int
    is_numpy_dtype: bool
    is_signed: bool


@dataclass(frozen=True)
class FloatType:
    """
    Used to aggregate numpy and python floating point types

    """

    bit_width: int
    is_numpy_dtype: bool
    is_signed: clscond = True


@dataclass(frozen=True)
class Cast(Expression):
    expr: ValueRef
    as_type: type

    @property
    def is_integral(self):
        return issubclass(self.as_type, numbers.Integral)

    @property
    def is_float(self):
        return issubclass(self.as_type, numbers.Real)

    @property
    def subexprs(self):
        yield self.expr
