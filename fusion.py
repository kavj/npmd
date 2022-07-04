import itertools

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import List, Set, Union

import ir

from analysis import count_clobbers
from errors import CompilerError
from symbol_table import SymbolTable
from traversal import depth_first_sequence_blocks, walk
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


def make_access_func(node: Union[ir.NameRef, ir.Expression]):
    if isinstance(node, ir.Subscript):
        if isinstance(node.index, ir.Slice):
            start, stop, step = node.index.subexprs
            return node.value, AccessFunc(start, stop, step)
        else:
            # slices can't be bound here, so this is unambiguous
            dimref = ir.SingleDimRef(node, ir.One)
            return node.value, AccessFunc(ir.Zero, dimref, ir.One)
    elif isinstance(node, ir.Expression):
        return None, AccessFunc(ir.Zero, ir.SingleDimRef(node, ir.Zero), ir.One)
    elif isinstance(node, ir.NameRef):
        return node, AccessFunc(ir.Zero, ir.SingleDimRef(node, ir.Zero), ir.One)
    else:
        msg = f'Cannot extract access func from "{node}".'
        raise CompilerError(msg)


def group_access_funcs(array_refs: Set[ir.NameRef, ir.Subscript]):
    # Now group array access functions by base
    access_funcs_by_name = defaultdict(set)
    for ref in array_refs:
        name, accessor = make_access_func(ref)
        access_funcs_by_name[name].add(accessor)
    return access_funcs_by_name


class ArrayExprCollector:

    def __init__(self, symbols: SymbolTable):
        self.symbols = symbols
        self.typer = TypeHelper(symbols)
        self.array_ops = set()
        self.seen = set()
        # also keep track of array refs to track access funcs
        self.array_refs = set()

    def __call__(self, node: ir.ValueRef):
        self.visit(node)

    def visit_bin_or_compare(self, node: Union[ir.BinOp, ir.CompareOp]):
        if node in self.seen:
            return
        self.seen.add(node)
        # Record cases where both terms are array references
        if self.typer.is_array(node.left) and self.typer.is_array(node.right):
            self.array_refs.add(node.left)
        for subexpr in node.subexprs:
            self.visit(subexpr)

    @singledispatchmethod
    def visit(self, node):
        msg = f'Array op collector is not implemented for "{node}"'
        raise TypeError(msg)

    @visit.register
    def _(self, node: ir.NameRef):
        if self.typer.is_array(node):
            self.array_refs.add(node)

    @visit.register
    def _(self, node: ir.Subscript):
        if self.typer.is_array(node):
            self.array_refs.add(node)

    @visit.register
    def _(self, node: ir.BinOp):
        self.visit_bin_or_compare(node)

    @visit.register
    def _(self, node: ir.CompareOp):
        self.visit_bin_or_compare(node)

    @visit.register
    def _(self, node: ir.Expression):
        if node in self.seen:
            return
        # it's possible to have something like 1 + (a + b).min()
        # due to weird nesting, so we have to assume anything might hide an array expression
        for subexpr in node.subexprs:
            self.visit(subexpr)

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


def find_array_expressions(stmts: List[ir.StmtBase], symbols: SymbolTable):
    collector = ArrayExprCollector(symbols)
    for stmt in itertools.chain(*depth_first_sequence_blocks(stmts)):
        if isinstance(stmt, ir.Assign):
            collector.visit(stmt.value)
            collector.visit(stmt.target)
        elif isinstance(stmt, (ir.InPlaceOp, ir.SingleExpr)):
            collector.visit(stmt.value)
        elif isinstance(stmt, ir.Return):
            if stmt.value is not None:
                collector.visit(stmt.value)
        elif isinstance(stmt, (ir.IfElse, ir.WhileLoop)):
            collector.visit(stmt.test)
        elif isinstance(stmt, ir.ForLoop):
            collector.visit(stmt.iterable)
    return collector.array_ops, collector.array_refs


def fuse_(block: List[ir.StmtBase], symbols: SymbolTable, ephemeral: Set[ir.NameRef]):
    """
    This is meant to run after locality optimizers.
    :param block:
    :param symbols:
    :param ephemeral:
    :return:
    """
    # gather array expressions
    array_exprs, array_refs = find_array_expressions(block, symbols)
    accessor_by_name = group_access_funcs(array_refs)

    # Now find those that appear together
    groups = []
    for expr in array_exprs:
        for group in groups:
            if not group.isdisjoint(expr.subexprs):
                group.update(expr.subexprs)
                break
        else:
            # doesn't match existing
            groups.append(set(expr.subexprs))
        # This is a bit more conservative than it needs to be.
        # Technically if write stride is faster than read stride, we're okay.

    counts = count_clobbers(block)

