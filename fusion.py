from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Iterable, List

import ir

from errors import CompilerError
from symbol_table import SymbolTable
from traversal import array_safe_walk, depth_first_sequence_statements, walk
from type_checks import TypeHelper


# Need a way to find reduction constraints
# Consider a[a != a.min()]
# this places a dependency on fully iterating over "a"

# Todo: Support where, abs, in, not in, trig funcs, complex multiply and divide
#       all, any


@dataclass(frozen=True)
class AccessFunc:
    start: ir.ValueRef
    stop: ir.ValueRef
    step: ir.ValueRef


class ArrayOpCollector:

    def __init__(self, symbols: SymbolTable):
        self.symbols = symbols
        self.array_ops = set()

    @singledispatchmethod
    def visit(self, node):
        msg = f'Array op collector is not implemented for "{node}"'
        raise TypeError(msg)

    @visit.register
    def _(self, node: ir.ValueRef):
        return False

    @visit.register
    def _(self, node: ir.Expression):
        if node in self.array_ops:
            return True
        for subexpr in node.subexprs:
            if isinstance(subexpr, ir.NameRef):
                if isinstance(self.symbols.check_type(subexpr), ir.ArrayType):
                    self.array_ops.add(node)
                    return True
            elif isinstance(subexpr, ir.Expression):
                if self.visit(subexpr):
                    return True
        return False

    @visit.register
    def _(self, node: ir.SELECT):
        if node in self.array_ops:
            return True
        # always check these to make sure an ambiguous case didn't slip through
        # it would be annoying to catch after this
        on_true_is_array = self.visit(node.on_true)
        on_false_is_array = self.visit(node.on_false)
        if on_true_is_array != on_false_is_array:
            msg = f'Select statement "{node}" is ambiguous'
            raise CompilerError(msg)
        # The result is only a blend if the predicate is an array, but the predicate may be an
        # array reduction to a scalar, which is truth tested as a predicate indicating a shallow
        # assignment binding the corresponding array to the expression result
        return self.visit(node.predicate) or on_true_is_array


class AccessFuncCollector:

    def __init__(self, symbols: SymbolTable):
        self.symbols = symbols
        self.accessors = defaultdict(set)

    @singledispatchmethod
    def visit(self, node):
        msg = f'No method to collect access funcs from "{node}"'
        raise TypeError(msg)

    @visit.register
    def _(self, node: ir.Expression):
        for subexpr in node.subexprs:
            self.visit(subexpr)

    @visit.register
    def _(self, node: ir.NameRef):
        return AccessFunc(start=ir.Zero, stop=ir.SingleDimRef(node, dim=ir.Zero), step=ir.One)

    @visit.register
    def _(self, node: ir.Subscript):
        # don't descend on value here
        name = node.value
        # ensure array type
        if not isinstance(name, ir.ArrayType):
            msg = f'Cannot subscript non-array type "{name}".'
            raise CompilerError(msg)
        if isinstance(node.index, ir.Slice):
            start, stop, step = node.index.subexprs
            if stop is None:
                stop = ir.SingleDimRef(name, ir.Zero)
            self.accessors[name].add(AccessFunc(start=start, stop=stop, step=step))
        else:
            base_array_type = self.symbols.check_type(name)
            reduced_type = base_array_type.without_leading_dim()
            if isinstance(reduced_type, ir.ArrayType):
                self.accessors[name].add(AccessFunc(start=ir.Zero, stop=ir.SingleDimRef(name, ir.One), step=ir.One))


def find_array_expressions(stmts: List[ir.StmtBase], symbols: SymbolTable):
    array_op_collector = ArrayOpCollector(symbols)
    for stmt in depth_first_sequence_statements(stmts):
        if isinstance(stmt, ir.Assign):
            array_op_collector.visit(stmt.value)
            array_op_collector.visit(stmt.target)
        elif isinstance(stmt, (ir.InPlaceOp, ir.SingleExpr)):
            array_op_collector.visit(stmt.value)
        elif isinstance(stmt, ir.Return):
            if stmt.value is not None:
                array_op_collector.visit(stmt.value)
        elif isinstance(stmt, (ir.IfElse, ir.WhileLoop)):
            array_op_collector.visit(stmt.test)
        elif isinstance(stmt, ir.ForLoop):
            array_op_collector.visit(stmt.iterable)
    return array_op_collector.array_ops


def fuse_statements(block: Iterable[ir.StmtBase]):
    access_funcs = defaultdict(set)
    chunks = []
    curr = []
    for stmt in block:
        # If this introduces an incompatible access pattern
        # check if curr has any statement.
        # If so, break it
        pass
