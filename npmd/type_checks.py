import networkx as nx
import numpy as np

from contextlib import contextmanager
from functools import singledispatchmethod
from typing import Any, List, Union

import npmd.ir as ir

from npmd.blocks import build_function_graph
from npmd.errors import CompilerError
from npmd.pretty_printing import PrettyFormatter
from npmd.symbol_table import SymbolTable
from npmd.traversal import walk_nodes
from npmd.utils import extract_name, unpack_iterated


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


class TypeHelper:

    def __init__(self, symbols: SymbolTable, allow_none=False):
        self.symbols = symbols
        self.format = PrettyFormatter()
        self.allow_none = allow_none

    def __call__(self, node):
        return self.infer(node)

    def promote_min_max(self, node: ir.ValueRef, types: List[Union[np.dtype, type, ir.ArrayType, ir.StringConst, ir.NoneRef]]):
        if any(isinstance(t, ir.ArrayType) for t in types):
            msg = f'Min operator is not applicable to array "{self.format(node)}"'
            raise CompilerError(msg)
        tentative = np.promote_types(*types)
        if tentative != types[0]:
            msg = f'Min and max require that we use the first type. ' \
                  f'This is only supported in cases where this agrees with promote types: "{self.format(node)}"'
            raise CompilerError(msg)
        return tentative

    @singledispatchmethod
    def raise_on_non_integer(self, node):
        msg = f'No method to integer check "{node}"'
        raise TypeError(msg)

    @raise_on_non_integer.register
    def _(self, node: ir.CONSTANT):
        return node.is_integer

    @raise_on_non_integer.register
    def _(self, node: ir.NameRef):
        t = self.infer(node)
        if not is_integer(t):
            msg = f'\"{node.name}\" is used as a parameter to a subscript index, slice parameter, enumerate, or ' \
                  f'range parameter, which must have an integer type. Received \"{t}\"'
            raise CompilerError(msg)

    @raise_on_non_integer.register
    def _(self, node: ir.Expression):
        t = self.infer(node)
        if not is_integer(t):
            msg = f'\"{self.format(node)}\" is used as a parameter to a subscript index, slice parameter, enumerate, ' \
                  f'or range parameter, which must have an integer type. Received \"{t}\"'
            raise CompilerError(msg)

    @property
    def index_type(self):
        return self.symbols.default_index_type

    @singledispatchmethod
    def infer(self, node):
        msg = f'No method to infer types for input "{node}"'
        raise TypeError(msg)

    @infer.register
    def _(self, node: ir.Call):
        # Todo: This needs to actually distinguish return types
        return ir.NoneRef

    @infer.register
    def _(self, node: ir.MIN):
        types = [self.infer(subexpr) for subexpr in node.subexprs]
        return self.promote_min_max(node, types)

    @infer.register
    def _(self, node: ir.MAX):
        types = [self.infer(subexpr) for subexpr in node.subexprs]
        return self.promote_min_max(node, types)

    @infer.register
    def _(self, node: ir.NameRef):
        known = self.symbols.check_type(node, self.allow_none)
        return known

    @infer.register
    def _(self, node: ir.Subscript):
        base_type = self.infer(node.value)
        if not isinstance(base_type, ir.ArrayType):
            formatted = self.format(node)
            msg = f'Subscripting is only supported on arrays, Received "{formatted}" with type "{base_type}".'
            raise CompilerError(msg)
        # ensure we have a valid index
        if isinstance(node.index, (ir.Slice, ir.TUPLE)):
            for subexpr in node.index.subexprs:
                self.raise_on_non_integer(subexpr)
        else:
            self.raise_on_non_integer(node.index)
        if isinstance(node.index, ir.Slice):
            dims = base_type.ndims
        elif isinstance(node.index, ir.TUPLE):
            dims = base_type.ndims - len(node.index.elements)
        else:
            dims = base_type.ndims - 1
        if dims < 0:
            formatted = self.format(node)
            msg = f'Expression "{formatted}" is over-subscripted by {base_type.ndims-dims} dimensions.'
            raise CompilerError(msg)
        elif dims == 0:
            return base_type.dtype
        return ir.ArrayType(dims, base_type.dtype)

    @infer.register
    def _(self, node: ir.BinOp):
        return np.promote_types(*(self.infer(subexpr) for subexpr in node.subexprs))

    @infer.register
    def _(self, node: ir.CompareOp):
        for subexpr in node.subexprs:
            if isinstance(subexpr, ir.StringConst):
                formatted = self.format(subexpr)
                msg = f'Strings are ineligible for comparison, received "{formatted}"'
                raise CompilerError(msg)
        return np.dtype('bool')

    @infer.register
    def _(self, node: ir.BoolOp):
        for subexpr in node.subexprs:
            self.infer(subexpr)
        return np.dtype('bool')

    @infer.register
    def _(self, node: ir.TRUEDIV):
        left, right = (self.infer(subexpr) for subexpr in node.subexprs)
        left_dtype = left if isinstance(left, np.dtype) else left.dtype
        right_dtype = right if isinstance(right, np.dtype) else right
        left_unity = left_dtype.type(1)
        right_unity = right_dtype.dtype(1)
        return type(left_unity / right_unity)

    @infer.register
    def _(self, node: ir.SingleDimRef):
        array_type = self.infer(node.base)
        if array_type.ndims < node.dim.value + 1:
            formatted = self.format(node.base)
            msg = f'Dimension {node.dim.value} is out of bounds for array {formatted} with total dimensions {array_type.ndims}'
            raise CompilerError(msg)
        return self.index_type

    @infer.register
    def _(self, node: ir.SELECT):
        if_type = self.infer(node.on_true)
        else_type = self.infer(node.on_false)
        if if_type != else_type:
            msg = f'If and else components of if expression have conflicting types "{if_type}" and "{else_type}"'
            raise CompilerError(msg)
        return if_type

    @infer.register
    def _(self, node: ir.CONSTANT):
        return node.dtype

    @infer.register
    def _(self, node: ir.StringConst):
        return ir.StringConst

    @infer.register
    def _(self, node: ir.NoneRef):
        return ir.NoneRef

    @singledispatchmethod
    def iterated_type(self, node):
        msg = f'No method to infer iterator output for expression "{node}".'
        raise TypeError(msg)

    @iterated_type.register
    def _(self, node: ir.AffineSeq):
        for subexpr in node.subexprs:
            self.raise_on_non_integer(subexpr)
        for subexpr in node.subexprs:
            self.infer(subexpr)
        return self.index_type

    @iterated_type.register
    def _(self, node: ir.ValueRef):
        base_type = self.infer(node)
        if isinstance(base_type, ir.ArrayType):
            if base_type.ndims == 1:
                return base_type.dtype
            else:
                return ir.ArrayType(base_type.ndims - 1, base_type.dtype)
        else:
            msg = f'Type "{base_type}" is not iterable.'
            raise CompilerError(msg)


class TypeInference:

    def __init__(self,
                 symbols: SymbolTable):
        self.symbols = symbols
        self.type_checker = TypeHelper(symbols)
        self.types_changed = False

    def bind_type(self, node: ir.NameRef, t: Any):
        existing_type = self.symbols.check_type(node, allow_none=True)
        if existing_type is None:
            self.symbols.bind_type(node, t)
            self.types_changed = True
        elif self.symbols.has_inferred_type(node):
            # with declared types but not inferred types, casts are allowed
            # this will raise on type conflict
            self.symbols.bind_type(node, t)

    @contextmanager
    def type_context(self):
        self.types_changed = False
        yield
        self.types_changed = False

    @singledispatchmethod
    def visit(self, node):
        msg = f'No method to infer types for "{node}".'
        raise TypeError(msg)

    @visit.register
    def _(self, node: ir.StmtBase):
        pass

    @visit.register
    def _(self, node: ir.Assign):
        value_type = self.type_checker.infer(node.value)
        changed = False
        if isinstance(node.target, ir.NameRef):
            self.bind_type(node.target, value_type)
        else:
            self.type_checker.infer(node.value)
        return changed

    @visit.register
    def _(self, node: ir.InPlaceOp):
        self.type_checker.infer(node.value)

    @visit.register
    def _(self, node: ir.SingleExpr):
        if not isinstance(node.value, ir.Call):
            self.type_checker.infer(node.value)

    @visit.register
    def _(self, node: ir.ForLoop):
        for target, iterable in unpack_iterated(node.target, node.iterable):
            target_type = self.type_checker.iterated_type(iterable)
            self.bind_type(target, target_type)

    @visit.register
    def _(self, node: ir.Return):
        self.type_checker.infer(node.value)

    @visit.register
    def _(self, node: ir.IfElse):
        self.type_checker.infer(node.test)

    @visit.register
    def _(self, node: ir.WhileLoop):
        self.type_checker.infer(node.test)


def validate_types(func: ir.Function, symbols: SymbolTable):
    type_checker = TypeHelper(symbols)
    for stmt in walk_nodes(func.body):
        if isinstance(stmt, (ir.Assign, ir.InPlaceOp)):
            type_checker.infer(stmt.value)
            type_checker.infer(stmt.target)
        elif isinstance(stmt, (ir.IfElse, ir.WhileLoop)):
            type_checker.infer(stmt.test)
        elif isinstance(stmt, (ir.SingleExpr, ir.Return)):
            type_checker.infer(stmt.value)
        elif isinstance(stmt, ir.ForLoop):
            for target, iterable in unpack_iterated(stmt.target, stmt.iterable):
                type_checker.iterated_type(iterable)
                type_checker.infer(target)


def infer_types(func: ir.Function, symbols: SymbolTable):
    type_infer = TypeInference(symbols)
    graph = build_function_graph(func)
    body_block, = graph.successors(graph.entry_block)
    changed = True
    while changed:
        with type_infer.type_context():
            for block in nx.dfs_preorder_nodes(graph.graph, body_block):
                for stmt in block:
                    type_infer.visit(stmt)
            changed = type_infer.types_changed
    # Now validate immediately, so that stuff doesn't get lost due to dce and the like
    validate_types(func, symbols)


def check_return_type(node: ir.Function, symbols: SymbolTable):
    typer = TypeHelper(symbols)
    return_types = set()
    for stmt in walk_nodes(node.body):
        if isinstance(stmt, ir.Return):
            if isinstance(stmt.value, ir.NoneRef):
                return_types.add(ir.NoneRef())
            else:
                return_types.add(typer(stmt.value))
    if len(return_types) > 1:
        msg = f"Function {extract_name(node)} has more than one return type, received {return_types}."
        raise CompilerError(msg)
    elif len(return_types) == 1:
        return_type = return_types.pop()
    else:
        return_type = ir.NoneRef()  # coerce to void at C level outside wrapper
    return return_type


dtype_pretty_lookup = {
    np.dtype('bool'): 'numpy.bool_',
    np.dtype('uint8'): 'numpy.uint8',
    np.dtype('int8'): 'numpy.int8',
    np.dtype('int32'): 'numpy.int32',
    np.dtype('int64'): 'numpy.int64',
    np.dtype('float32'): 'numpy.float64',
    np.dtype('float64'): 'numpy.float64',
    np.dtype('complex64'): 'numpy.complex64',
    np.dtype('complex128'): 'numpy.complex128'
}


def dump_symbol_type_info(symbols: SymbolTable):
    print(f'Function: {symbols.namespace} arguments\n')
    for name in symbols.arguments:
        t = symbols.check_type(name)
        if isinstance(t, ir.ArrayType):
            dtype_formatted = dtype_pretty_lookup[t.dtype]
            formatted_type = f'numpy.ndarray[{dtype_formatted}]'
        else:
            formatted_type = dtype_pretty_lookup[t]
        name_to_type = f'{name.name}: {formatted_type}'
        print(name_to_type)


def invalid_loop_iterables(node: ir.ForLoop, symbols: SymbolTable):
    type_checker = TypeHelper(symbols)
    for _, iterable in unpack_iterated(node.target, node.iterable):
        if not isinstance(iterable, ir.AffineSeq):
            t = type_checker(iterable)
            if not isinstance(t, ir.ArrayType):
                yield iterable
