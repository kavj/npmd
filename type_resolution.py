import numpy as np
import ir

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
