import numbers
import typing
from dataclasses import dataclass

binaryops = {"+", "-", "*", "/", "//", "%", "**", "<<", ">>", "|", "^", "&", "@"}
inplace_ops = {"+=", "-=", "*=", "/=", "//=", "%=", "**=", "<<=", ">>=", "|=", "^=", "&=", "@="}
unaryops = {"+", "-", "~", "not"}
boolops = {"and", "or"}
compareops = {"==", "!=", "<", "<=", ">", ">=", "is", "isnot", "in", "notin"}

supported_builtins = {'iter', 'range', 'enumerate', 'zip', 'all', 'any', 'max', 'min', 'abs', 'pow',
                      'round', 'reversed'}


@dataclass(frozen=True)
class Position:
    line_begin: int
    line_end: int
    col_begin: int
    col_end: int


clscond = typing.ClassVar[bool]


class StmtBase:
    is_control_flow: clscond = False  # any control flow construct
    is_scope_entry: clscond = False  # Function
    is_simple_entry: clscond = False  # ForLoop, WhileLoop, IfElse, does not alter scope
    is_conditional_branch: clscond = False  # IfElse statement
    is_loop_entry: clscond = False  # ForLoop, WhileLoop
    is_terminator: clscond = False  # Continue, Break, Return

    is_assign: clscond = False  # Assign, CascadeAssign
    may_assign: clscond = False  # Assign, CascadeAssign, ForLoop


class ObjectBase(typing.Protocol):
    is_expr: clscond = False  # expression node, never a value but encountered in the same places
    is_subscript: clscond = False  # actual subscript node
    is_constant: clscond = False  # literal constant
    is_array: clscond = False  # array or view
    is_counter: clscond = False  # range, enumerate


class Expression(ObjectBase, typing.Protocol):
    """
    This is the start of a base class for expressions.
    
    For our purposes, we distinguish an expression from a value by asking whether it can be 
    an assignment target. 

    The term 'expression_like' is used elsewhere in the code to include mutable targets, where
    assignment alters a data structure as opposed to simply binding a value to a name.

    """

    is_expr: clscond = True

    @property
    def subexprs(self):
        raise NotImplementedError

    @property
    def is_simple(self):
        return not any(e.is_expr for e in self.subexprs)


class Constant(ObjectBase, typing.Protocol):
    """
    Base class for anything that can be the target of an assignment. 

    """
    is_constant: clscond = True

    @property
    def unwrapped(self):
        return self if not self.is_constant else self.value


@dataclass(frozen=True)
class AttributeRef(ObjectBase):
    """
    Limited number of attributes are read only.

    """
    value: ObjectBase
    attr: typing.Tuple[str]


@dataclass(frozen=True)
class BoolNode(Constant):
    value: bool
    is_constant: clscond = True


@dataclass(frozen=True)
class EllipsisNode(Constant):
    # singleton class
    value: typing.Any = Ellipsis
    is_constant: clscond = True


@dataclass(frozen=True)
class FloatNode(Constant):
    value: numbers.Real
    is_constant: clscond = True


@dataclass(frozen=True)
class IntNode(Constant):
    value: numbers.Integral
    is_constant: clscond = True


@dataclass(frozen=True)
class StringNode(Constant):
    value: str
    is_constant: clscond = True


# Top Level

@dataclass(frozen=True)
class NameRef(ObjectBase):
    # variable name ref
    name: typing.Union[str, AttributeRef]


@dataclass(frozen=True)
class Argument(ObjectBase):
    name: NameRef
    annot: str = None
    commt: str = None
    defaultvalue: ObjectBase = None


@dataclass(frozen=True)
class ArrayType(ObjectBase):
    # This is necessary to easily describe the type of an otherwise anonymous subscript,
    # particularly in cases of subscripts that are not bound to explicit array references.
    ndims: int
    dtype: type


@dataclass(frozen=True)
class ArrayRef(ObjectBase):
    atype: ArrayType
    base: typing.Optional[ObjectBase] = None
    # assume contiguous unless specified otherwise due to striding in
    # array creation expression or input typing parameters
    is_contiguous: bool = True
    is_array: clscond = True

    def __post_init__(self):
        assert (self.dims is None or (self.ndims == len(self.dims)))

    @property
    def is_view(self):
        return self.base is not None

    @property
    def ndims(self):
        return self.atype.ndims

    @property
    def dtype(self):
        return self.atype.dtype


@dataclass(frozen=True)
class ShapeRef(ObjectBase):
    array: ArrayRef

    @property
    def ndims(self):
        return self.array.ndims


@dataclass
class Function:
    name: str
    args: typing.List[Argument]
    body: typing.List[StmtBase]
    types: typing.List[type]
    arrays: typing.List[ArrayRef]
    is_scope_entry: clscond = True


@dataclass
class Module:
    """
    This aggregates imports and Function graphs

    """

    funcs: typing.List[Function]
    imports: typing.List[typing.Any]
    is_scope_entry: clscond = True


@dataclass(frozen=True)
class SimpleSlice(Expression):
    """
    IR representation of a slice.

    Per dimension constraints can be applied

    """

    start: typing.Optional[ObjectBase]
    stop: typing.Optional[ObjectBase]
    step: typing.Optional[ObjectBase]

    def __post_init__(self):
        assert (isinstance(self.start, tuple) and isinstance(self.stop, tuple) and isinstance(self.step, tuple))

    @property
    def subexprs(self):
        yield self.start
        yield self.stop
        yield self.step


@dataclass(frozen=True)
class Unsupported(Expression):
    name: str
    msg: str


@dataclass(frozen=True)
class Subscript(Expression):
    value: ObjectBase
    slice: ObjectBase
    is_subscript: clscond = True

    @property
    def subexprs(self):
        yield self.value
        yield self.slice

    @property
    def is_single_index(self):
        return isinstance(self.slice, (NameRef, IntNode))


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

    elements: typing.Tuple[ObjectBase, ...]

    def __post_init__(self):
        assert (isinstance(self.elements, tuple))

    @property
    def subexprs(self):
        for e in self.elements:
            yield e


Targetable = typing.TypeVar('Targetable', NameRef, Subscript, Tuple)


@dataclass(frozen=True)
class BinOp(Expression):
    left: ObjectBase
    right: ObjectBase
    op: str

    @property
    def subexprs(self):
        yield self.left
        yield self.right

    @property
    def in_place(self):
        return self.op in inplace_ops


@dataclass(frozen=True)
class BoolOp(Expression):
    """
    Boolean operation using a single logical operation and an arbitrary
    number of operands. 

    Comparisons between expressions are compared without modification.

    """

    operands: typing.Tuple[ObjectBase, ...]
    op: str

    def __post_init__(self):
        assert (isinstance(self.operands, tuple))

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
    args: typing.Tuple[ObjectBase, ...]
    # expressions must be safely hashable, so we can't use a dictionary here
    keywords: typing.Tuple[typing.Tuple[str, ObjectBase], ...]

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
class CompareOp(Expression):
    """
    Comparison reduction

    """

    operands: typing.Tuple[ObjectBase, ...]
    ops: typing.Tuple[str, ...]

    def __post_init__(self):
        assert (isinstance(self.operands, tuple) and isinstance(self.ops, tuple))

    @property
    def subexprs(self):
        for operand in self.operands:
            yield operand


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

    start: ObjectBase
    stop: typing.Optional[ObjectBase]
    step: ObjectBase

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

    test: ObjectBase
    if_expr: ObjectBase
    else_expr: ObjectBase

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

    iterable: ObjectBase
    is_iterator_like: clscond = True

    @property
    def subexprs(self):
        yield self.iterable


@dataclass(frozen=True)
class UnaryOp(Expression):
    operand: ObjectBase
    op: str

    @property
    def subexprs(self):
        yield self.operand


@dataclass(frozen=True)
class Zip(Expression):
    """
    High level sentinel representing a zip object. This is the only unpackable iterator type in this IR.
    Enumerate(object) is handled by Zip(Counter, object).

    """

    elements: typing.Tuple[ObjectBase, ...]

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
    value: ObjectBase
    pos: Position
    # Record optional annotation, since we need to check
    # reachability before applying.
    annot: str = None
    is_assign: clscond = True
    may_assign: clscond = True

    @property
    def in_place(self):
        return isinstance(self.value, BinOp) and self.value.in_place

    @property
    def is_permutation(self):
        return (isinstance(self.target, Tuple)
                and isinstance(self.value, Tuple)
                and len(self.target.elements) == len(self.value.elements))


@dataclass
class SingleExpr(StmtBase):
    expr: ObjectBase
    pos: Position


@dataclass
class Break(StmtBase):
    pos: Position
    is_terminator: clscond = True


@dataclass
class CascadeAssign(StmtBase):
    """ 
    a = b = c

    Subscripts are unsupported, because they have the potential
    to create dependency chains.

    """

    targets: typing.List[Targetable]
    value: ObjectBase
    pos: Position
    is_assign: clscond = True
    may_assign: clscond = True


@dataclass
class Continue(StmtBase):
    pos: Position
    is_terminator: clscond = True


@dataclass
class ForLoop(StmtBase):
    iterable: ObjectBase
    target: ObjectBase
    body: typing.List[StmtBase]
    pos: Position
    is_control_flow: clscond = True
    is_loop_entry: clscond = True
    may_assign: clscond = True


@dataclass
class IfElse(StmtBase):
    test: ObjectBase
    if_branch: typing.List[StmtBase]
    else_branch: typing.List[StmtBase]
    pos: Position
    is_control_flow: clscond = True
    is_conditional_branch: clscond = True

    @property
    def empty(self):
        return not (self.if_branch or self.else_branch)

    @property
    def empty_if(self):
        return len(self.if_branch) == 0

    @property
    def empty_else(self):
        return len(self.else_branch) == 0


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
    value: typing.Optional[ObjectBase]
    pos: Position
    is_terminator: clscond = True


@dataclass
class WhileLoop(StmtBase):
    test: ObjectBase
    body: typing.List[StmtBase]
    pos: Position
    is_loop_entry: clscond = True
    is_control_flow: clscond = True


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
    expr: ObjectBase
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
