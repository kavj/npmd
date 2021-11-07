import itertools

import numpy as np

from functools import singledispatchmethod

import ir

from errors import CompilerError
from visitor import ExpressionVisitor

# Todo: At the tree layer, this should be much with fine grained tests moving to dataflow layer.
#       In particular, just check for bad use of division and truth testing of arrays


Float32 = ir.ScalarType(bits=32, integral=False, boolean=False)
Float64 = ir.ScalarType(bits=64, integral=False, boolean=False)
Int32 = ir.ScalarType(bits=32, integral=True, boolean=False)
Int64 = ir.ScalarType(bits=64, integral=True, boolean=False)
Predicate32 = ir.ScalarType(bits=32, integral=True, boolean=True)
Predicate64 = ir.ScalarType(bits=64, integral=True, boolean=True)
FPredicate32 = ir.ScalarType(bits=32, integral=False, boolean=True)
FPredicate64 = ir.ScalarType(bits=64, integral=False, boolean=True)
BoolType = ir.ScalarType(bits=8, integral=True, boolean=True)

DefaultInt = Int32 if np.int_(0).itemsize == 4 else Int64

# defaults, can be overridden
by_input_type = {np.int32: Int32,
                 np.int64: Int64,
                 np.float32: Float32,
                 np.float64: Float64,
                 bool: BoolType,
                 np.bool_: BoolType}

by_ir_type = {Int32: np.int32,
              Int64: np.int64,
              Float32: np.float32,
              Float64: np.float64,
              BoolType: np.bool}

by_input_type_name = {"numpy.int32": Int32,
                      "numpy.int64": Int64,
                      "numpy.float32": Float32,
                      "numpy.float64": Float64,
                      "numpy.bool": BoolType,
                      "numpy.bool_": BoolType}

np_func_by_binop = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.true_divide,
    "//": np.floor_divide,
    "%": np.mod,
    "**": np.power,
    "<<": np.left_shift,
    ">>": np.right_shift,
    "|": np.bitwise_or,
    "&": np.bitwise_and,
    "^": np.bitwise_xor,
    "~": np.bitwise_not
}


# initially supported, untyped ints and other ranges require additional
# work, and they are less commonly used
scalar_types = {np.int32, np.int64, np.float32, np.float64, np.bool}

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


inplace_to_ooplace = {
    "+=": "+",
    "-=": "-",
    "*=": "*",
    "/": "/",
    "//=": "//",
    "%=": "%",
    "**=": "**",
    "<<=": "<<",
    ">>=": ">>",
    "|=": "|",
    "&=": "&",
    "^=": "^",
    "~=" : "~"
}


def get_compare_type(a, b):
    # will be updated
    bits = max(a.bits, b.bits)
    integral = a.integral and b.integral
    if integral and bits == 32:
        return Predicate32
    elif integral and bits == 64:
        return Predicate64
    elif (not integral) and bits == 32:
        return FPredicate32
    elif (not integral) and bits == 64:
        return FPredicate64


def resolve_binop_type(a, b, op):
    if op in bitwise:
        # bitwise requires integral or predicate
        if a.boolean and b.boolean:
            return get_compare_type(a,b)
    a_ = by_ir_type[a]
    b_ = by_ir_type[b]
    if op == "/":
        # get numpy
        a_one = a_(1)
        b_one = b_(1)
        c_one = np.divide(a_one, b_one)
        c_ = type(c_one)
        c = by_input_type.get(c_)
        if c is None:
            msg = f"No op to implement division for types {a_} and {b_}."
            raise CompilerError(msg)
        return c
    res_ = np.result_type(a_, b_)
    res = by_input_type[res_]
    return res


def type_from_spec(bit_width, is_integral, is_boolean):
    """
    Return a type object from interned types.
    """
    if bit_width == 8:
        if is_integral:
            if is_boolean:
                return BoolType
        msg = "The only currently supported 8 bit format is integer boolean."
        raise ValueError(msg)
    if bit_width not in (32, 64):
        msg = "Only 32 and 64 bit numeric data types are supported."
        raise ValueError(msg)
    if is_integral:
        if bit_width == 64:
            return Predicate64 if is_boolean else Int64
        else:
            return Predicate32 if is_boolean else Int32
    elif bit_width == 64:
        return FPredicate64 if is_boolean else Float64
    else:
        return FPredicate32 if is_boolean else Float32


def merge_truth_types(types):
    assert len(types) > 0
    bit_width = 0
    is_integral = True
    for t in types:
        if not t.boolean:
            t = truth_type_from_type(t)
        bit_width = max(bit_width, t.bits)
        is_integral &= t.is_integral
    if bit_width == 8:
        if not is_integral:
            raise TypeError
    elif bit_width == 32:
        return Predicate32 if is_integral else FPredicate32
    elif bit_width == 64:
        return Predicate64 if is_integral else FPredicate64
    else:
        msg = f"Unsupported bit width {bit_width}."
        raise CompilerError(msg)


class ExprTypeInfer(ExpressionVisitor):
    """
    This exists to determine the output types generated by expressions. Actual inference
    injects greater ambiguity here.

    references:
    Ole Ageson, The Cartesian Product Algorithm
    Simple and Precise Type Inference of Parametric Polymorphism

    """

    def __init__(self, symbols):
        # types updated externally
        self.symbols = symbols
        self.expr_types = None

    def __call__(self, expr):
        assert isinstance(expr, ir.ValueRef)
        return self.visit(expr)

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Subscript):
        array_type = self.visit(node.value)
        if not isinstance(array_type, ir.ArrayType):
            # would be better not to raise here..
            msg = f"Cannot subscript non-array type {array_type}."
            raise CompilerError(msg)
        if isinstance(node.slice, ir.Slice):
            t = array_type
        else:
            # single index
            ndims = array_type.ndims - 1
            if ndims == 0:
                t = array_type.dtype
            else:
                t = ir.ArrayType(ndims, array_type.dtype)
        return t

    @visit.register
    def _(self, node: ir.NameRef):
        sym = self.symbols.lookup(node)
        return sym.type_

    @visit.register
    def _(self, node: ir.BinOp):
        # no caching since they may change
        left = self.visit(node.left)
        right = self.visit(node.right)
        expr_type = binops_dispatch.get((left, right))
        if expr_type is None:
            msg = f"No signature match for operator {node.op} with candidate signatures: ({left}, {right})."
            raise CompilerError(msg)
        return expr_type

    @visit.register
    def _(self, node: ir.UnaryOp):
        return self.visit(node.operand)

    @visit.register
    def _(self, node: ir.CompareOp):
        # Todo: currently assumes all scalar types... need to generalize
        ltype = self.visit(node.left)
        rtype = self.visit(node.right)
        left_dtype = ltype.dtype if isinstance(ltype, ir.ArrayType) else ltype
        right_dtype = rtype.dtype if isinstance(ltype, ir.ArrayType) else rtype
        if isinstance(left_dtype, BoolType) and isinstance(right_dtype, BoolType):
            cmp_type = BoolType
        else:
            cmp_type = get_compare_type(left_dtype, right_dtype)
        return cmp_type

    @visit.register
    def _(self, node: ir.BoolOp):
        truth_types = []
        for operand in node.subexprs:
            type_ = self.visit(operand)
            if isinstance(type_, ir.ArrayType):
                msg = f"Cannot truth test array type {operand}."
                return TypeMismatch(msg)
            truth_types.append(truth_type_from_type(type_))
        return merge_truth_types(truth_types)
