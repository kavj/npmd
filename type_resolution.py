import itertools

import numpy as np
import ir


from errors import CompilerError

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

by_input_type_name = {"numpy.int32": Int32,
                      "numpy.int64": Int64,
                      "numpy.float32": Float32,
                      "numpy.float64": Float64,
                      "numpy.bool": BoolType,
                      "numpy.bool_": BoolType}

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

# no support for arbitrary arithmetic on predicate or plain boolean types

binops_dispatch = {
    (Int32, Int32): Int32,
    (Int32, Int64): Int64,
    (Int32, Float32): Float64,
    (Int32, Float64): Float64,
    (Int64, Int32): Int64,
    (Int64, Int64): Int64,
    (Int64, Float32): Float64,
    (Int64, Float64): Float64,
    (Float32, Int32): Float64,
    (Float32, Int64): Float64,
    (Float32, Float32): Float64,
    (Float32, Float64): Float64,
    (Float64, Int32): Float64,
    (Float64, Int64): Float64,
    (Float64, Float32): Float64,
    (Float64, Float64): Float64
}

cmp_dispatch = {
    (Int32, Int32): Predicate32,
    (Int32, Int64): Predicate64,
    (Int32, Float32): FPredicate32,
    (Int32, Float64): FPredicate64,
    (Int64, Int32): Predicate32,
    (Int64, Int64): Predicate64,
    (Int64, Float32): FPredicate64,
    (Int64, Float64): FPredicate64,
    (Float32, Int32): FPredicate32,
    (Float32, Int64): FPredicate64,
    (Float32, Float32): FPredicate32,
    (Float32, Float64): FPredicate64,
    (Float64, Int32): Float64,
    (Float64, Int64): Float64,
    (Float64, Float32): Float64,
    (Float64, Float64): Float64
}

true_div_dispatch = {
    (Int32, Int32): Float64,
    (Int32, Int64): Float64,
    (Int32, Float32): Float64,
    (Int32, Float64): Float64,
    (Int64, Int32): Float64,
    (Int64, Int64): Float64,
    (Int64, Float32): Float64,
    (Int64, Float64): Float64,
    (Float32, Int32): Float64,
    (Float32, Int64): Float64,
    (Float32, Float32): Float32,
    (Float32, Float64): Float64,
    (Float64, Int32): Float64,
    (Float64, Int64): Float64,
    (Float64, Float32): Float64,
    (Float64, Float64): Float64
}


bitwise_dispatch = {
    (Int32, Int32): Int32,
    (Int32, Int64): Int64,
    (Int64, Int32): Int64,
    (Int64, Int64): Int64
}


dispatch = {
    "+": binops_dispatch,
    "-": binops_dispatch,
    "*": binops_dispatch,
    "//": binops_dispatch,
    "%": binops_dispatch,
    "**": binops_dispatch,
    "/": true_div_dispatch,
    "<<": bitwise_dispatch,
    ">>": bitwise_dispatch,
    "|": bitwise_dispatch,
    "&": bitwise_dispatch,
    "^": bitwise_dispatch,
    "+=": binops_dispatch,
    "-=": binops_dispatch,
    "*=": binops_dispatch,
    "//=": binops_dispatch,
    "%=": binops_dispatch,
    "**=": binops_dispatch,
    "/=": true_div_dispatch,
    "<<=": bitwise_dispatch,
    ">>=": bitwise_dispatch,
    "|=": bitwise_dispatch,
    "&=": bitwise_dispatch,
    "^=": bitwise_dispatch,
}


def truth_type_from_type(base_type):
    """
    Truth cast interned type to interned truth type.
    """
    assert isinstance(base_type, ir.ScalarType)
    if base_type.bits == 32:
        return Predicate32 if base_type.integral else FPredicate32
    elif base_type.bits == 64:
        return FPredicate32 if base_type.integral else FPredicate64
    else:
        msg = f"Unknown type {base_type}."
        raise ValueError(msg)


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


def resolve_binop_type(left_type, right_type, op):
    assert op in itertools.chain(ir.binary_ops, ir.in_place_ops)
    # if utils.is_multiplication()
