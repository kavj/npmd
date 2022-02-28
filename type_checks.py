import numpy as np

from functools import singledispatchmethod

import ir

from errors import CompilerError
from symbol_table import symbol_table
from utils import extract_name, get_stmt_types


def check_return_type(node: ir.Function, symbols: symbol_table):
    stmts = get_stmt_types(node.body, (ir.Return,))
    typer = TypeHelper(symbols)
    return_types = {typer.check_type(stmt.value) for stmt in stmts if stmt.value is not None}
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
    def __init__(self, syms: symbol_table):
        self.symbols = syms

    @singledispatchmethod
    def check_type(self, expr):
        msg = f"No method to check type for {expr}"
        raise NotImplementedError(msg)

    @check_type.register
    def _(self, expr: ir.NameRef):
        return self.symbols.check_type(expr)

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
    def _(self, expr: ir.Constant):
        return expr.dtype

    @check_type.register
    def _(self, expr: ir.BinOp):
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
        return result_type

    @check_type.register
    def _(self, expr: ir.Subscript):
        value, index = expr.subexprs
        value_type = self.check_type(value)
        if not isinstance(value_type, ir.ArrayType):
            msg = f"Cannot subscript non-array type {value_type}."
            raise CompilerError(msg)
        if isinstance(index, ir.Tuple):
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
                msg =  f"Non integer index, type: {subtype}."
                raise CompilerError(msg)
        if value_type.ndims < ndims_removed:
            msg = f"Over subscripted array {expr}."
            raise CompilerError(msg)
        if ndims_removed == value_type.ndims:
            return value_type.dtype
        else:
            return ir.ArrayType(ndims=value_type.ndims-ndims_removed, dtype=value_type.dtype)

    @check_type.register
    def _(self, expr: ir.UnaryOp):
        return self.check_type(expr.operand)

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
    def _(self, expr: ir.Constant):
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
                prefix = 'i'
        t = self.check_type(like_expr)
        s = self.symbols.make_unique_name_like(prefix, t)
        return s
