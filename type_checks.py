import numpy as np

from functools import singledispatchmethod
from typing import Any, List, Optional, Tuple

import ir

from errors import CompilerError
from symbol_table import SymbolTable
from utils import extract_name, unpack_iterated

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


def contains_stmt_types(stmts: List[ir.StmtBase], stmt_types: Tuple[Any, ...]):
    for stmt in stmts:
        if any(isinstance(stmt, stmt_type) for stmt_type in stmt_types):
            return True
        elif isinstance(stmt, (ir.ForLoop, ir.WhileLoop)):
            if type(stmt) in stmt_types:
                return True
            elif contains_stmt_types(stmt.body, stmt_types):
                return True
        elif isinstance(stmt, ir.IfElse):
            if ir.IfElse in stmt_types:
                return True
            elif contains_stmt_types(stmt.if_branch, stmt_types) or contains_stmt_types(stmt.else_branch, stmt_types):
                return True
    return False


def get_stmt_types(stmts: List[ir.StmtBase], stmt_types: Tuple[Any, ...]):
    retrievals = []
    for stmt in stmts:
        if any(isinstance(stmt, stmt_type) for stmt_type in stmt_types):
            retrievals.append(stmt)
        if isinstance(stmt, (ir.ForLoop, ir.WhileLoop)):
            retrievals.extend(get_stmt_types(stmt.body, stmt_types))
        elif isinstance(stmt, ir.IfElse):
            retrievals.extend(get_stmt_types(stmt.if_branch, stmt_types))
            retrievals.extend(get_stmt_types(stmt.else_branch, stmt_types))
    return retrievals


def check_return_type(node: ir.Function, symbols: SymbolTable):
    stmts = get_stmt_types(node.body, (ir.Return,))
    typer = TypeHelper(symbols)
    return_types = set()
    for stmt in stmts:
        if stmt.value is None:
            return_types.add(ir.NoneRef())
        else:
            return_types.add(typer.check_type(stmt.value))
    if len(return_types) > 1:
        msg = f"Function {extract_name(node)} has more than one return type, received {return_types}."
        raise CompilerError(msg)
    elif len(return_types) == 1:
        return_type = return_types.pop()
    else:
        return_type = ir.NoneRef()  # coerce to void at C level outside wrapper
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
    def __init__(self, syms: SymbolTable, allow_none=False, default_prefix: Optional[str] = None):
        self.symbols = syms
        self.allow_none = allow_none
        self.default_prefix = default_prefix if default_prefix is not None else "i"

    @singledispatchmethod
    def check_type(self, expr):
        msg = f"No method to check type for {expr}"
        raise NotImplementedError(msg)

    @check_type.register
    def _(self, expr: ir.ArrayInitializer):
        ndims = len(expr.shape)
        dtype = expr.dtype
        return ir.ArrayType(ndims, dtype)

    @check_type.register
    def _(self, expr: ir.NameRef):
        t = self.symbols.check_type(expr, self.allow_none)
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
    def _(self, expr: ir.StringConst):
        return type(expr)

    @check_type.register
    def _(self, expr: ir.CONSTANT):
        return expr.dtype

    @check_type.register
    def _(self, expr: ir.NoneRef):
        return expr

    @check_type.register
    def _(self, expr: ir.BinOp):
        left, right = expr.subexprs
        left_type = self.check_type(left)
        right_type = self.check_type(right)
        if left_type is None or right_type is None:
            return
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
        if value_type is None:
            return
        if not isinstance(value_type, ir.ArrayType):
            msg = f"Cannot subscript non-array type {value_type}."
            raise CompilerError(msg)
        if isinstance(index, ir.TUPLE):
            # check that all integer types
            subtypes = [self.check_type(subexpr) for subexpr in index.subexprs]
            if None in subtypes:
                return
            elif any(t not in (np.int32, np.int64) for t in subtypes):
                msg = f"Non integer indices, types: {subtypes}."
                raise CompilerError(msg)
            # now check sizing..
            ndims_removed = 2
        else:
            ndims_removed = 1
            subtype = self.check_type(index)
            if subtype is None:
                return
            elif subtype not in (np.int32, np.int64):
                msg = f"Non integer index, type: {subtype}."
                raise CompilerError(msg)
        if value_type.ndims < ndims_removed:
            msg = f"Over subscripted array {expr}."
            raise CompilerError(msg)
        if ndims_removed == value_type.ndims:
            return value_type.dtype
        else:
            arrtype = ir.ArrayType(ndims=value_type.ndims - ndims_removed, dtype=value_type.dtype)
            return arrtype

    @check_type.register
    def _(self, expr: ir.UnaryOp):
        t = self.check_type(expr.operand)
        return t

    def check_dtype(self, expr: ir.ValueRef):
        t = self.check_type(expr)
        if isinstance(t, ir.ArrayType):
            t = t.dtype
        return t

    def is_array(self, node: ir.ValueRef):
        t = self.check_type(node)
        return isinstance(t, ir.ArrayType)

    def is_scalar(self, node: ir.ValueRef):
        t = self.check_type(node)
        return isinstance(t, np.dtype)

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


class TypeInference:
    """
    This differs from type helper, in that it actively assigns a type. It does this by checking
    inferred expression and trying to bind to that, then raising an error on conflict. Conflicts
    do get resolved with casts if assigning to something with an unambiguous type.
    """

    def __init__(self, symbols: SymbolTable):
        self.symbols = symbols
        self.typer = TypeHelper(symbols)

    @singledispatchmethod
    def visit(self, node):
        msg = f'No type inference method for node "{node}".'
        raise TypeError(msg)

    @visit.register
    def _(self, node: ir.Function):
        for arg in node.args:
            if self.symbols.check_type(arg) is None:
                msg = f'Missing argument type for "{arg}".'
                raise CompilerError(msg)
        self.visit(node.body)

    @visit.register
    def _(self, node: list):
        for stmt in node:
            self.visit(stmt)

    @visit.register
    def _(self, node: ir.StmtBase):
        pass

    @visit.register
    def _(self, node: ir.Assign):
        if isinstance(node.target, ir.NameRef):
            t = self.typer.check_type(node.value)
            target_sym = self.symbols.lookup(node.target)
            if target_sym.type_ is None or target_sym.type_is_inferred:
                self.symbols.bind_type(node.target, t)

    @visit.register
    def _(self, node: ir.IfElse):
        self.visit(node.if_branch)
        self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.WhileLoop):
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.ForLoop):
        for target, iterable in unpack_iterated(node.target, node.iterable):
            assert isinstance(target, ir.NameRef)
            target_sym = self.symbols.lookup(target)
            if target_sym.type_ is None or target_sym.type_is_inferred:
                if isinstance(iterable, ir.AffineSeq):
                    self.symbols.bind_type(target, np.dtype("int32"))
                else:
                    iterable_type = self.typer.check_type(iterable)
                    if isinstance(iterable_type, ir.ArrayType):
                        ndims = iterable_type.ndims - 1
                        dtype = iterable_type.dtype
                        if ndims != 0:
                            t = ir.ArrayType(ndims, dtype)
                        else:
                            t = dtype
                        self.symbols.bind_type(target, t)
                    else:
                        raise CompilerError("bad type for iterable")
        self.visit(node.body)
