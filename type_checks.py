import numpy as np

from functools import singledispatchmethod
from typing import Optional

import ir

from errors import CompilerError
from symbol_table import SymbolTable
from utils import extract_name, get_stmt_types


int_dtypes = {
    np.dtype('int8'),
    np.dtype('int16'),
    np.dtype('int32'),
    np.dtype('int64'),
    np.dtype('uint8'),
    np.dtype('uint16'),
    np.dtype('uint32'),
    np.dtype('uint64'),
}


real_float_dtypes = {
    np.dtype('float32'),
    np.dtype('float64'),
}


complex_dtypes = {
    np.dtype('complex64'),
    np.dtype('complex128')
}


float_dtypes = real_float_dtypes.union(complex_dtypes)

real_dtypes = int_dtypes.union(real_float_dtypes)


dtype_to_suffix = {np.dtype('int8'): 's8',
                   np.dtype('int16'): 's16',
                   np.dtype('int32'): 's32',
                   np.dtype('int64'): 's64',
                   np.dtype('uint8'): 'u8',
                   np.dtype('uint16'): 'u16',
                   np.dtype('uint32'): 'u32',
                   np.dtype('uint64'): 'u64',
                   np.dtype('float32'): 'f32',
                   np.dtype('float64'): 'f64',
                   np.dtype('complex64'): 'f32',  # corresponding real component type for ops
                   np.dtype('complex128'): 'f64'}  # that don't require unpacking


def is_integer(dtype: np.dtype):
    return dtype in int_dtypes


def is_float(dtype: np.dtype):
    return dtype in float_dtypes


def is_real(dtype: np.dtype):
    return dtype in real_dtypes


def check_return_type(node: ir.Function, symbols: SymbolTable):
    stmts = get_stmt_types(node.body, (ir.Return,))
    typer = TypeHelper(symbols)
    return_types = set()
    for stmt in stmts:
        if stmt.value is None:
            return_types.add(None)
        else:
            return_types.add(typer.check_type(stmt.value))
    # should check if all paths terminate..
    # eg either ends in a return statement or every branch has a return statement
    if len(return_types) > 1:
        msg = f"Function {extract_name(node)} has more than one return type, received {return_types}."
        raise CompilerError(msg)
    elif len(return_types) == 1:
        return_type = return_types.pop()
    else:
        return_type = None  # coerce to void at C level outside wrapper
    return return_type


def check_truediv_dtype(left: np.dtype, right: np.dtype):
    unity_left = left.type(1)
    unity_right = right.type(1)
    return (unity_left / unity_right).dtype


def check_binop_dtype(left: np.dtype, right: np.dtype, op: ir.BinOp):
    if isinstance(op, ir.TRUEDIV):
        dtype = check_truediv_dtype(left, right)
    else:
        dtype = np.result_type(left, right)
    return dtype


class TypeHelper:
    def __init__(self, syms: SymbolTable, default_prefix: Optional[str] = None):
        self.symbols = syms
        self.default_prefix = default_prefix if default_prefix is not None else "i"
        self._cached = {}

    @singledispatchmethod
    def check_type(self, expr):
        msg = f"No method to check type for {expr}"
        raise NotImplementedError(msg)

    @check_type.register
    def _(self, expr: ir.NameRef):
        if expr in self._cached:
            # don't use get here, as it won't catch invalid Nones
            return self._cached[expr]
        t = self.symbols.check_type(expr)
        self._cached[expr] = t
        return t

    @check_type.register
    def _(self, expr: ir.AffineSeq):
        # annoying, because not portable, think of something later
        return np.dtype(np.int_)

    @check_type.register
    def _(self, expr: ir.CompareOp):
        # should validate..
        return ir.bool_type

    @check_type.register
    def _(self, expr: ir.BoolOp):
        return ir.bool_type

    @check_type.register
    def _(self, expr: ir.CAST):
        # cast nodes should have a make cas with more checking
        return expr.target_type

    @check_type.register
    def _(self, expr: ir.CONSTANT):
        return expr.dtype

    @check_type.register
    def _(self, expr: ir.BinOp):
        if expr in self._cached:
            return self._cached[expr]
        left, right = expr.subexprs
        left_type = self.check_type(left)
        right_type = self.check_type(right)
        left_is_array = isinstance(left_type, ir.ArrayType)
        right_is_array = isinstance(right_type, ir.ArrayType)
        if left_is_array and right_is_array:
            # Todo: not always correct..
            result_ndims = max(left_type.ndims, right_type.ndims)
            result_dtype = check_binop_dtype(left_type.dtype, right_type.dtype, expr)
            result_type = ir.ArrayType(result_ndims, result_dtype)
        elif left_is_array:
            result_dtype = check_binop_dtype(left_type.dtype, right, expr)
            result_type = ir.ArrayType(left.ndims, result_dtype)
        elif right_is_array:
            result_dtype = check_binop_dtype(left_type, right_type.dtype, expr)
            result_type = ir.ArrayType(right_type.ndims, result_dtype)
        else:
            result_type = check_binop_dtype(left_type, right_type, expr)
        self._cached[expr] = result_type
        return result_type

    @check_type.register
    def _(self, expr: ir.Subscript):
        if expr in self._cached:
            return self._cached[expr]
        value, index = expr.subexprs
        value_type = self.check_type(value)
        if not isinstance(value_type, ir.ArrayType):
            msg = f"Cannot subscript non-array type {value_type}."
            raise CompilerError(msg)
        if isinstance(index, ir.TUPLE):
            # check that all integer types
            subtypes = [self.check_type(subexpr) for subexpr in index.subexprs]
            if any(t not in (np.int32, np.int64) for t in subtypes):
                msg = f"Non integer indices, types: {subtypes}."
                raise CompilerError(msg)
            # now check sizing..
            ndims_removed = 2
        else:
            ndims_removed = 1
            subtype = self.check_type(index)
            if subtype not in (np.int32, np.int64):
                msg = f"Non integer index, type: {subtype}."
                raise CompilerError(msg)
        if value_type.ndims < ndims_removed:
            msg = f"Over subscripted array {expr}."
            raise CompilerError(msg)
        if ndims_removed == value_type.ndims:
            self._cached[expr] = value_type.dtype
            return value_type.dtype
        else:
            arrtype = ir.ArrayType(ndims=value_type.ndims - ndims_removed, dtype=value_type.dtype)
            self._cached[expr] = arrtype
            return arrtype

    @check_type.register
    def _(self, expr: ir.UnaryOp):
        if expr in self._cached:
            return self._cached[expr]
        t = self.check_type(expr.operand)
        self._cached[expr] = t
        return t

    def check_dtype(self, expr: ir.ValueRef):
        t = self.check_type(expr)
        if isinstance(t, ir.ArrayType):
            t = t.dtype
        return t

    @singledispatchmethod
    def is_predicate(self, expr):
        raise NotImplementedError

    @is_predicate.register
    def _(self, expr: ir.NameRef):
        return self.check_dtype(expr) == ir.bool_type

    @is_predicate.register
    def _(self, expr: ir.CONSTANT):
        return expr.dtype == ir.bool_type

    @is_predicate.register
    def _(self, expr: ir.Subscript):
        # this is not a predicate if it's predicated, only if its dtype matches
        return self.is_predicate(expr.value)

    @is_predicate.register
    def _(self, expr: ir.CompareOp):
        return True

    @is_predicate.register
    def _(self, expr: ir.BoolOp):
        return True

    def declare_typed(self, type_, prefix: Optional[str] = None):
        if prefix is None:
            prefix = self.default_prefix
        return self.symbols.make_unique_name_like(prefix, type_)

    def declare_like(self, like_expr, prefix=None):
        """
        Returns a name that like_expr can safely bind to
        :param like_expr:
        :param prefix:
        :return:
        """
        if prefix is None:
            if isinstance(like_expr, ir.NameRef):
                prefix = like_expr
            else:
                prefix = self.default_prefix
        t = self.check_type(like_expr)
        s = self.symbols.make_unique_name_like(prefix, t)
        return s
