import numbers
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import singledispatchmethod

import numpy as np

import ir
from visitor import walk_multiple, AssignCollector

# initially supported, untyped ints and other ranges require additional
# work, and they are less commonly used
scalar_types = {int, float, np.float32, np.float64, np.int32, np.int64}

binary_ops = {"+", "-", "*", "/", "//", "%", "**", "<<", ">>", "|", "^", "&",
              "+=", "-=", "*=", "/=", "//=", "%=", "**="}

bitwise_ops = {"<<", ">>", "|", "&", "^", "<<=", ">>=", "|=", "&=", "^="}

matmul_ops = {"@", "@="}

truediv_ops = {"/", "/="}

unary_ops = {"+", "-", "~", "not"}

bool_ops = {"and", "or"}

compare_ops = {"==", "!=", "<", "<=", ">", ">=", "is", "isnot", "in", "notin"}

supported_builtins = {'iter', 'range', 'enumerate', 'zip', 'all', 'any', 'max', 'min', 'abs', 'pow',
                      'round', 'reversed'}

binops = {"+", "-", "*", "//", "%", "**"}
binops_inplace = {"+=", "-=", "*=", "//=", "%=", "**="}
div = {"/"}
div_inplace = {"/="}
matmul = {"@"}
matmul_inplace = {"@="}
bitwise = {"<<", ">>", "|", "&", "^"}
bitwise_inplace = {"<<=", ">>=", "|=", "&=", "^="}

# Python's normal dispatching rules make for potentially inefficient SIMD code. It's possible to solve around them
# in a lot of cases, but they may be sensitive to type perturbations as a result of minor assignment differences.
# Instead we're assigning rules meant to unify integer variables to 32 or 64 bit mode with the same treatment applied
# to floating point. By default, for now, we're disallowing downcast on write. Efficient simd upcast on read is usually
# provided by the ISA. Some of these things can impact register pressure, so we'll need to provide some hooks to
# manipulate unroll factor used by vectorization.


binops_dispatch = {
    (int, int): int,
    (int, float): float,
    (int, np.int32): np.int64,
    (int, np.int64): np.int64,
    (int, np.float32): np.float64,
    (int, np.float64): np.float64,

    (float, int): float,
    (float, float): float,
    (float, np.int32): np.float64,
    (float, np.int64): np.float64,
    (float, np.float32): np.float64,
    (float, np.float64): np.float64,

    (np.int32, int): np.int64,
    (np.int32, float): np.float64,
    (np.int32, np.int32): np.int32,
    (np.int32, np.int64): np.int64,
    (np.int32, np.float32): np.float64,
    (np.int32, np.float64): np.float64,

    (np.int64, int): np.int64,
    (np.int64, float): np.float64,
    (np.int64, np.int32): np.int64,
    (np.int64, np.int64): np.int64,
    (np.int64, np.float32): np.float64,
    (np.int64, np.float64): np.float64,

    (np.float32, int): np.float64,
    (np.float32, float): np.float64,
    (np.float32, np.int32): np.float64,
    (np.float32, np.int64): np.float64,
    (np.float32, np.float32): np.float32,
    (np.float32, np.float64): np.float64,

    (np.float64, int): np.float64,
    (np.float64, float): np.float64,
    (np.float64, np.int32): np.float64,
    (np.float64, np.int64): np.float64,
    (np.float64, np.float32): np.float64,
    (np.float64, np.float64): np.float64,
}

# This extends Numpy's rules for inplace array operations to also apply to scalars.
# This means that inplace operations cannot apply type promotion.

binops_inplace_dispatch = {
    (int, int),
    (int, np.int32),
    (int, np.int64),

    (float, int),
    (float, float),
    (float, np.int32),
    (float, np.int64),
    (float, np.float32),
    (float, np.float64),

    (np.int32, np.int32),

    (np.int64, int),
    (np.int64, np.int32),
    (np.int64, np.int64),

    (np.float32, np.float32),

    (np.float64, int),
    (np.float64, float),
    (np.float64, np.int32),
    (np.float64, np.int64),
    (np.float64, np.float32),
    (np.float64, np.float64),
}

div_dispatch = {
    (int, int): float,
    (int, float): float,
    (int, np.int32): np.float64,
    (int, np.int64): np.float64,
    (int, np.float32): np.float64,
    (int, np.float64): np.float64,

    (float, int): float,
    (float, float): float,
    (float, np.int32): np.float64,
    (float, np.int64): np.float64,
    (float, np.float32): np.float64,
    (float, np.float64): np.float64,

    (np.int32, int): np.float64,
    (np.int32, float): np.float64,
    (np.int32, np.int32): np.float64,
    (np.int32, np.int64): np.float64,
    (np.int32, np.float32): np.float64,
    (np.int32, np.float64): np.float64,

    (np.int64, int): np.float64,
    (np.int64, float): np.float64,
    (np.int64, np.int32): np.float64,
    (np.int64, np.int64): np.float64,
    (np.int64, np.float32): np.float64,
    (np.int64, np.float64): np.float64,

    (np.float32, int): np.float64,
    (np.float32, float): np.float64,
    (np.float32, np.int32): np.float64,
    (np.float32, np.int64): np.float64,
    (np.float32, np.float32): np.float32,
    (np.float32, np.float64): np.float64,

    (np.float64, int): np.float64,
    (np.float64, float): np.float64,
    (np.float64, np.int32): np.float64,
    (np.float64, np.int64): np.float64,
    (np.float64, np.float32): np.float64,
    (np.float64, np.float64): np.float64,
}

div_inplace_dispatch = {

    (float, int),
    (float, float),
    (float, np.int32),
    (float, np.int64),
    (float, np.float32),
    (float, np.float64),

    (np.float32, np.float32),

    (np.float64, int),
    (np.float64, float),
    (np.float64, np.int32),
    (np.float64, np.int64),
    (np.float64, np.float32),
    (np.float64, np.float64),
}

bitwise_dispatch = {
    (int, int): int,
    (int, np.int32): np.int64,
    (int, np.int64): np.int64,
    (np.int32, int): np.int64,
    (np.int32, np.int32): np.int32,
    (np.int32, np.int64): np.int64,
    (np.int64, int): np.int64,
    (np.int64, np.int32): np.int64,
    (np.int64, np.int64): np.int64,
}

bitwise_inplace_dispatch = {
    (int, int),
    (int, np.int32),
    (int, np.int64),
    (np.int32, np.int32),
    (np.int64, int),
    (np.int64, np.int32),
    (np.int64, np.int64),
}

dispatch = {
    "+": binops_dispatch,
    "-": binops_dispatch,
    "*": binops_dispatch,
    "//": binops_dispatch,
    "%": binops_dispatch,
    "**": binops_dispatch,
    "/": div_dispatch,
    "<<": bitwise_dispatch,
    ">>": bitwise_dispatch,
    "|": bitwise_dispatch,
    "&": bitwise_dispatch,
    "^": bitwise_dispatch,
    "+=": binops_inplace_dispatch,
    "-=": binops_inplace_dispatch,
    "*=": binops_inplace_dispatch,
    "//=": binops_inplace_dispatch,
    "%=": binops_inplace_dispatch,
    "**=": binops_inplace_dispatch,
    "/=": div_inplace_dispatch,
    "<<=": bitwise_inplace_dispatch,
    ">>=": bitwise_inplace_dispatch,
    "|=": bitwise_inplace_dispatch,
    "&=": bitwise_inplace_dispatch,
    "^=": bitwise_inplace_dispatch,
}


def unify_type(types):
    if ir.ArrayType in types:
        if len(types) > 1:
            return
        return next(iter(types))
    typeiter = iter(types)
    prev = next(typeiter)
    t = None
    for t in typeiter:
        # unify using standard binop type dispatching
        prev, t = t, prev
        t = binops_dispatch.get((prev, t))
        if t is None:
            return
    return t


def is_negative_number(node):
    if not node.is_constant:
        return False
    if isinstance(node, (ir.IntNode, ir.FloatNode)):
        return node.value < 0
    return False


class TypedNameRef:

    def __init__(self):
        self.types = set()
        self.assigned_to = set()
        self.is_iterable = False

    @property
    def may_grow(self):
        # type system will require cleanup to allow for bottom monomorphic type
        # to be determined by context
        return float not in self.types and np.float64 not in self.types

    @property
    def type(self):
        return unify_type(self.types)

    def add_type(self, t):
        self.types.add(t)


class TypeBase(ABC):

    @abstractmethod
    def add_type(self, t):
        raise NotImplementedError

    @property
    @abstractmethod
    def type(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def may_grow(self):
        raise NotImplementedError


class TypedSubscript(TypeBase):

    def __init__(self, is_single_index):
        self.types = set()
        self.assigned_to = set()
        self.is_single_index = is_single_index

    @abstractmethod
    def add_type(self, t):
        self.types.add(t)

    @property
    def type(self):
        if len(self.types) == 1:
            t = next(iter(self.types))
            if not isinstance(t, ir.ArrayType):
                return TypedInvalidRef(self, msg="Subscripting non-array types is unsupported.")
            if self.is_single_index:
                if t.ndims == 1:
                    return t.dtype
                else:
                    return ir.ArrayType(t.ndims - 1, t.dtype)
            else:
                return t
        return TypedInvalidRef(self, msg="Subscripted datatype is not monomorphic.")

    @property
    def may_grow(self):
        return False


class TypedExpr:

    def __init__(self, type_dispatcher):
        # it would be better if this checked by type
        # or better still if dispatching referred to module or context state
        self.type_dispatcher = type_dispatcher
        self.types = defaultdict(set)
        self.assigned_to = set()

    def matching_type(self, t):
        """
        get all signatures matching this type
        """

        return tuple(self.types[t]) if t in self.types else ()

    def add_type(self, sig):
        """
        Simple add type. Query .type before and after to detect changes in unified type.
        """
        t = self.type_dispatcher.get(sig)
        self.types[t].add(sig)

    @property
    def type(self):
        return unify_type(self.types)

    @property
    def may_grow(self):
        # this is primarily aimed at numeric types
        return float not in self.types and np.float64 not in self.types


class TypedInvalidRef:
    """
    what to include...
    reason?
    generating expression?
    some kind of info..

    """

    def __init__(self, node, msg=""):
        self.node = node
        self.msg = msg


def unpack_expressions(exprs):
    """
    Expands all expressions, yielding

    top: a set of expressions, which are not sub-expressions of any other
    params: the free variable parameters required by each expression
    expr_set: the set of all expressions, expanded to include sub-expressions

    """
    params = {}
    subexprs = set()
    expr_set = set()
    for expr in walk_multiple(exprs):
        if not expr.is_expr:
            continue
        expr_set.add(expr)
        p = set()
        for subexpr in expr.subexprs:
            if subexpr.is_expr:
                subexprs.add(subexpr)
                p.update(params.get(subexpr))
            else:
                p.add(subexpr)
        params[expr] = p
    top = expr_set.difference(subexprs)
    expr_set.difference_update(top)
    top = {t: params[t] for t in top}
    subexprs = {s: params[s] for s in expr_set}
    return top, subexprs


class Typer:

    def __init__(self, typed, call_typing=None):
        self.input_types = typed
        self.typed = {}
        # track used by where we actually have expressions rather than type info
        self.users = defaultdict(set)
        self.call_typing = call_typing if call_typing is not None else {}

    @singledispatchmethod
    def infer_type(self, node):
        if node in self.typed:
            return self.typed.get(node).type
        elif isinstance(node, (ir.BoolOp, ir.BoolNode, ir.CompareOp)):
            tn = None
            self.typed[node] = tn
            return bool
        elif isinstance(node, ir.IntNode):
            tn = None
            self.typed[node] = tn
            return int
        elif isinstance(node, ir.FloatNode):
            tn = None
            self.typed[node] = tn
            return float
        else:
            raise NotImplementedError(f"cannot find type for node: {node}")

    @infer_type.register
    def _(self, node: ir.Subscript):
        # get base array type
        if node in self.typed:
            return self.typed.get(node).type
        v = self.infer_type(node.value)
        if not isinstance(v, ir.ArrayType):
            return TypedInvalidRef(node, f"Cannot subscript type: {v}")
        ndims = v.ndims
        s = self.infer_type(node.slice)
        if isinstance(s, TypedInvalidRef):
            return TypedInvalidRef(node, "invalid slice")
        else:
            # Check if single index, otherwise dims remains unchanged
            t = TypedSubscript(ir.ArrayType(ndims - 1, v.dtype))
            self.typed[node] = t
            return t

    @infer_type.register
    def _(self, node: ir.SimpleSlice):
        if node in self.typed:
            return self.typed[node]
        for subexpr in node.subexprs:
            if subexpr not in self.typed:
                t = self.infer_type(subexpr)
                if isinstance(t, TypedInvalidRef) or not issubclass(t.type, numbers.Integral):
                    return TypedInvalidRef(node, msg="invalid slice component: " + t.msg)

    @infer_type.register
    def _(self, node: ir.NameRef):
        if node in self.input_types:
            return self.input_types[node]
        elif node in self.typed:
            return self.input_types[node].type
        else:
            return TypedInvalidRef(node, "No type for node")

    @infer_type.register
    def _(self, node: ir.Call):
        # These need to be registered prior to this point.
        # These should receive fixed signatures that may be used
        # as sentinels for typing. It also makes type constraints
        # better behaved.
        t = self.call_typing.get(type(node))
        if t is None:
            t = TypedInvalidRef(node, "No type for call")
        return t

    @infer_type.register
    def _(self, node: ir.UnaryOp):
        # reasonable outside of unsigned types
        if node in self.typed:
            t = self.typed[node].type
        elif node.op == 'not':
            t = None  # TypedScalar(bool)
            self.typed[node] = t
            return t
        else:
            t = self.infer_type(node.operand)
            self.typed[node] = t
        return t

    @infer_type.register
    def _(self, node: ir.BinOp):
        if node in self.typed:
            return self.typed[node].type
        left = self.infer_type(node.left)
        right = self.infer_type(node.right)
        if isinstance(left, TypedInvalidRef):
            return TypedInvalidRef(node, msg=f"invalid left operand type")
        elif isinstance(right, TypedInvalidRef):
            return TypedInvalidRef(node, msg=f"invalid right operand type")
        dispatcher = dispatch.get(node.op)
        entry = TypedExpr(dispatcher)
        entry.add_type((left, right))
        self.typed[node] = entry
        return entry.type

    def infer_iterated(self, iterable):
        """
        Infer output type of iterating over this node
        """
        basetype = self.infer_type(iterable)
        if isinstance(basetype, TypedInvalidRef):
            return basetype
        elif isinstance(basetype, ir.ArrayType):
            ndims = basetype.ndims
            if ndims == 1:
                return basetype.dtype
            else:
                return ir.ArrayType(ndims - 1, basetype.dtype)
        elif isinstance(iterable, ir.Counter):
            pass
            # return TypedScalar(int)
        return TypedInvalidRef(iterable, f"Non-iterable type {basetype}")

    def update_user_types(self, v):
        for u in self.users.get(v):
            type_var = self.typed.get(u)
            t = tuple(self.typed.get(s).type for s in u.subexprs)
            type_var.add_type(t)

    def register(self, assign):
        target = assign.target
        value = assign.value
        if assign.iterated:
            vtype = self.infer_iterated(assign.value)
        else:
            # Ignore whether this is a subscript, and check what types are assigned to it.
            vtype = self.infer_type(value)

        # make this generic flow more complete..
        # Right now it has to assume a type variable exists, and they don't
        # have perfect matching interfaces (could be fixed somewhat)
        target_type_ref = self.typed.get(target)
        target_type_ref.add_type(vtype)
        # If get doesn't return one, we need to add it
        # may be better to factor out creation of these
        # self.typed[target] = target_type_ref
        self.update_user_types(target)


def compute_variable_types(entry, input_types):
    ac = AssignCollector()
    assigns, truth_tested, return_values = ac(entry)
    typer = Typer(input_types)
    for assign in assigns:
        typer.register(assign)
    # need input signatures for validation
    # get unified types
