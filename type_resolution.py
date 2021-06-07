import numpy as np
import typing

from collections import defaultdict
from functools import singledispatchmethod

import ir
from type_parsing import parse_array_create
from visitor import VisitorBase

# Todo: At the tree layer, this should be much with fine grained tests moving to dataflow layer.
#       In particular, just check for bad use of division and truth testing of arrays

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

# This rejects cases that require integer to float conversion,
# since they violate static typing rules.
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


class TypeCanonicalizer:

    def __init__(self):
        self.types = {}

    def get_canonical_type(self, initial_type):
        """
        Placeholder
        This is meant to aggregate compatible types into single types in the case of aliasing and for cases
        where we merely approximate the original type.

        """
        t = self.types.get(initial_type)
        return t if t is not None else initial_type


class TypeChecker(VisitorBase):
    """
    This only enforces types on explicit assignment. Compound expressions take on monomorphic types
    based on expression evaluation. Casts are only applied on explicit assignment.
    """

    def __call__(self, entry, vartypes: typing.Dict):
        self.vartypes = vartypes
        self.invalid_truth_tests = set()
        self.array_mismatches = defaultdict(set)
        self.expr_types = {}
        self.missing = set()
        self.visit(entry)
        missing = self.missing
        expr_types = self.expr_types
        array_mismatches = self.array_mismatches
        invalid_truth_tests = self.invalid_truth_tests
        self.missing = self.expr_types = self.array_mismatches = self.invalid_truth_tests = None
        return missing, expr_types, array_mismatches, invalid_truth_tests

    def lookup_type(self, var_or_expr):
        if isinstance(var_or_expr, ir.NameRef):
            t = self.vartypes.get(var_or_expr)
            if t is None:
                self.missing.add(var_or_expr)
        else:
            t = self.expr_types.get(var_or_expr)
        return t

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.NameRef):
        if node not in self.vartypes:
            self.missing.add(node)

    @visit.register
    def _(self, node: ir.Call):
        return parse_array_create(node)

    @visit.register
    def _(self, node: ir.Assign):
        self.visit(node.value)
        if isinstance(node.target, ir.Expression):
            self.visit(node.target)
        else:
            assigned_type = self.visit(node.target)
            expr_type = self.visit(node.value)
            # check for invalid casts
            if assigned_type != expr_type:
                if (isinstance(assigned_type, (ir.ArrayRef, ir.ViewRef))
                        or isinstance(expr_type, (ir.ArrayRef, ir.ViewRef))):
                    self.array_mismatches.add(node.target)

    @visit.register
    def _(self, node: ir.BinOp):
        # Todo: expand for boolean and compare ops
        if node in self.missing:
            return
        dispatcher = dispatch.get(node.op)
        lt = self.lookup_type(node.left)
        if lt is None:
            self.visit(node.left)
            lt = self.visit(node.left)
        rt = self.lookup_type(node.right)
        if rt is None:
            self.visit(node.right)
            rt = self.lookup_type(node.right)
        if lt is None or rt is None:
            self.missing.add(node)
        else:
            t = dispatcher.get((lt, rt))
            if t is None:
                self.missing.add(node)
            else:
                self.expr_types[node] = t

    @visit.register
    def _(self, node: ir.ForLoop):
        for target, iterable in node.walk_assignments():
            self.visit(target)
            self.visit(iterable)
            t = self.lookup_type(iterable)
            if t is None:
                self.missing.add(t)
                return
            if isinstance(t, ir.ArrayRef):
                target_type = self.lookup_type(target)
                if target_type is None:
                    self.array_mismatches[iterable].add(target)
                    return
                if t.ndims > 1:
                    if not isinstance(target_type, ir.ArrayRef):
                        self.array_mismatches[iterable].add(target)
                    elif target_type.ndims != t.ndims - 1:
                        self.array_mismatches[iterable].add(target)
                else:
                    self.array_mismatches[iterable].add(target)

    @visit.register
    def _(self, node: ir.WhileLoop):
        self.visit(node.test)
        t = self.lookup_type(node.test)
        if isinstance(t, ir.ArrayRef):
            self.invalid_truth_tests.add(t)
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.IfElse):
        self.visit(node.test)
        t = self.lookup_type(node.test)
        if isinstance(t, ir.ArrayRef):
            self.invalid_truth_tests.add(t)
        self.visit(node.if_branch)
        self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.IfExpr):
        self.visit(node.test)
        t = self.lookup_type(node.test)
        if isinstance(t, ir.ArrayRef):
            self.invalid_truth_tests.add(t)
        self.visit(node.if_expr)
        self.visit(node.else_expr)

    @visit.register
    def _(self, node: ir.Subscript):
        self.visit(node.value)
        self.visit(node.slice)
        existing = self.lookup_type(node)
        reduce_dims = not isinstance(node.slice, ir.Slice)
        t = self.lookup_type(node.value)
        if t is None:
            self.missing.add(node)
            return
        if reduce_dims:
            if isinstance(t, (ir.ArrayRef, ir.ViewRef)):
                if t.ndims > 1:
                    t = ir.ViewRef(node.value, node.slice)
                else:
                    t = node.value.dtype
            else:
                # over subscripted
                self.array_mismatches.add(node)
        if existing is None:
            self.expr_types[node] = t
        else:
            if existing != t:
                self.array_mismatches.add(node)
