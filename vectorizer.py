from collections import defaultdict
from functools import singledispatch, singledispatchmethod
from typing import Dict, Optional, Set
from type_checks import TypeHelper
import numpy as np

import ir
from errors import CompilerError
from symbol_table import symbol_table
from utils import contains_loops, unpack_iterated, extract_name
from visitor import walk, StmtVisitor


# find shared access patterns
# eg, suppose we have:   d = a + b
#                        e = a + c
# this shares access in "a" if dimensions are equal in both cases

# we also have dependent cases

#  d = a + b
#  e =  2 * d

# find escaping, eg arrays which are either arguments modified in place or return values


class AssignCollector(StmtVisitor):
    """
    Low level checker, it's assumed that only assign nodes generate views
    """

    def __init__(self):
        self.assigned = defaultdict(set)

    @classmethod
    def collect(cls, node):
        visitor = cls()
        visitor.visit(node)
        return visitor.assigned

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def visit(self, node: ir.Assign):
        self.assigned[node.target].add(node.value)

    @visit.register
    def _(self, node: ir.ForLoop):
        for target, value in unpack_iterated(node.target, node.iterable):
            self.assigned[target].add(value)


class ReadBeforeWriteCheck(StmtVisitor):
    """
    Used for further analysis

    This checks what variable names are involved in writes directly or subscripted and what nodes
    are involved in each.

    This is important when it comes to identifying cases where a node is reading but may also be written
    indirectly through an alias.

    For example, we can track whether an array is both read and written over the same scope
    and whether it's read in any before being read.

    """
    def __init__(self, writes, reads, read_before_written):
        self.written = writes
        self.read = reads
        self.read_before_written = read_before_written

    @classmethod
    def reentrant_check(cls, written=None, read=None, read_before_written=None):
        if written is None:
            written = set()
        if read is None:
            read = set()
        if read_before_written is None:
            read_before_written = set()
        visitor = cls(written, read, read_before_written)
        return visitor.read_before_written, visitor.read, visitor.written

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.ForLoop):
        for target, value in unpack_iterated(node.target, node.iterable):
            # this is likely to map affine expressions to names, but it's okay here
            self.read.extend(walk(value))
            target_name = ir.NameRef(extract_name(target))
            if target_name in self.read and target_name not in self.written:
                self.read_before_written.add(target_name)
            self.written.add(value)

    @visit.register
    def _(self, node: ir.IfElse):
        self.read.extend(walk(node.test))

    @visit.register
    def _(self, node: ir.Assign):
        self.read.extend(walk(node.value))
        if isinstance(node.target, ir.Subscript):
            name = ir.NameRef(extract_name(node.target))
            # don't flag this as a read before write unless the write precedes this statement
            if name in self.read and name not in self.written:
                self.read_before_written.add(name)
            self.written.add(name)
            self.read.extend(walk(node.target))
        elif isinstance(node.target, ir.NameRef):
            if node.target in self.read and node.target not in self.written:
                self.read_before_written.add(node.target)
                self.written.add(node.target)
        else:
            msg = f"Unsupported assignment {node}."
            raise CompilerError(msg)


def filter_array_assigns(target_to_values: Dict[ir.ValueRef, Set[ir.ValueRef]], type_helper: TypeHelper):
    view_assigns = {}
    for target, values in target_to_values.items():
        type_ = type_helper.check_type(target)
        if isinstance(type_, ir.ArrayType):
            view_assigns[target] = values
    return view_assigns


def analyze_func_vectorize(func: ir.Function, symbols: symbol_table, step_sizes: Dict[ir.NameRef, int]):
    target_to_values = AssignCollector.collect(func)
    args = set(func.args)
    type_helper = TypeHelper(symbols)
    # find assigned arrays
    array_assigns = filter_array_assigns(target_to_values, type_helper)


def find_loop_conflicts(node: ir.ForLoop, symbols: symbol_table):
    """
    Checks a loop for conflicts. May recurse..
    :param node:
    :param symbols:
    :return:
    """
    # this means incremented variable inside a loop (no induction optimization at the moment)
    # checking for writes that are not uniquely parameterized by index

    # look for nested loops and calls
    target_to_values = AssignCollector.collect(node)

    # check for read before written..
    read_before_write, read, written = ReadBeforeWriteCheck.reentrant_check(node)

    # check for reads and writes on the same array types with differing expressions


def find_max_loop_body_live_count(node: ir.ForLoop):
    """
    Checks loop live count. This is useful in estimating whether a downstream compiler is likely to
    spill generate spill code if we aim for latency hiding here.

    :param node:
    :return:
    """
    pass


def find_escaping_values(node: ir.Function):
    """
    This is a stub for an extremely naive escape analysis. Basically a scan for whether anything could possibly
    escape along loop bounds.

    :param node:
    :return:
    """

    pass


def find_max_block_live_count(node: list):
    """
    If this is a simple function, we might need to track within function scope

    :param node:
    :return:
    """
    pass


def check_if_vectorizable(node: ir.Function, symbols: symbol_table):
    """
    This checks a possibly nested call. It determines whether the structure favors vectorizing a single call
    vs vectorizing across calls.

    The former works well when:
         this function lacks nested loops and calls
         all memory here is accessed in a vectorizable pattern

    Vectorizing at the call level and transposing as necessary is preferred when:
         We have a step of zero or one between consecutive call iterations.

    It's a toss up when we have things like reductions where the ordering of iterations would differ
    if vectorized in the innermost function unless it's something where altering the order isn't even allowed.



    :param node:
    :param symbols:
    :return:
    """

    pass


class ValueLedger:
    def __init__(self, name_to_expr: Optional[Dict[ir.NameRef, ir.ValueRef]] = None,
                 expr_to_name: Optional[Dict[ir.Expression, ir.NameRef]] = None):
        self.name_to_expr = {} if name_to_expr is None else name_to_expr
        self.expr_to_name = {} if expr_to_name is None else expr_to_name

    def make_copy(self, copy_names: bool, copy_exprs: bool):
        name_to_expr = self.name_to_expr.copy() if copy_names else None
        expr_to_name = self.expr_to_name.copy() if copy_exprs else None
        return ValueLedger(name_to_expr, expr_to_name)

    @property
    def names(self):
        yield from self.name_to_expr

    @property
    def exprs(self):
        yield from self.expr_to_name

    @singledispatchmethod
    def bind(self, target, value):
        msg = f"No method to handle binding to type {type(target)}, target: {target}."
        raise TypeError(msg)

    @bind.register
    def _(self, target: ir.NameRef, value: ir.ValueRef):
        self.name_to_expr[target] = value

    @bind.register
    def _(self, target: ir.Expression, value: ir.NameRef):
        if target not in self.expr_to_name:
            msg = f"Cannot re-bind expression {target}."
            raise CompilerError(msg)
        self.expr_to_name[target] = value

    @singledispatchmethod
    def current_value(self, key):
        raise TypeError

    @current_value
    def _(self, key: ir.NameRef) -> ir.ValueRef:
        return self.name_to_expr.get(key, key)

    @current_value
    def _(self, key: ir.Expression) -> Optional[ir.NameRef]:
        # still important to check that the name reference remains bound to this
        return self.expr_to_name.get(key)

    @singledispatchmethod
    def rewrite_single(self, node):
        assert isinstance(node, ir.ValueRef)
        return node

    @rewrite_single.register
    def _(self, node: ir.NameRef):
        return self.current_value(node)

    @rewrite_single.register
    def _(self, node: ir.Expression):
        return node.reconstruct(*(self.rewrite_single(subexpr) for subexpr in node.subexprs))

    def rewrite(self, node: ir.ValueRef):
        assert isinstance(node, ir.ValueRef)
        if not isinstance(node, ir.Expression):
            return self.current_value(node)
        renamed = {}
        for expr in walk(node):
            existing = renamed.get(expr)
            if existing is None:
                existing = self.rewrite_single(expr)
                renamed[expr] = existing
        return node.reconstruct(*(renamed[subexpr] for subexpr in node.subexprs))


class uniformity_checker:
    """
    This is used to determine whether expressions may vary across iteratioins of a loop
    with known clobbers.
    """

    def __init__(self, clobbers):
        self.clobbers = clobbers

    def is_uniform(self, node: ir.ValueRef):
        return self.clobbers.isdisjoint(walk(node))


def contains_subscript(expr: ir.Expression):
    return any(isinstance(subexpr, ir.Subscript) for subexpr in walk(expr))


@singledispatch
def rewrite_stmt(stmt, symbols, accountant):
    raise TypeError


@rewrite_stmt.register
def _(stmt: ir.Assign, symbols: symbol_table, accountant: ValueLedger):
    value = accountant.rewrite(stmt.value)
    if isinstance(stmt.target, ir.NameRef):
        target = symbols.make_alias(stmt.target)
        # bind original to renamed
        accountant.bind(stmt.target, target)
        # bind renamed expression to alias
        accountant.bind(value, target)
    else:
        target = accountant.rewrite(stmt.target)
    return ir.Assign(target, value, stmt.pos)


@rewrite_stmt.register
def _(stmt: ir.InPlaceOp, symbols: symbol_table, accountant: ValueLedger):
    # rename if scalar
    value = accountant.rewrite(stmt.expr)
    if isinstance(stmt.target, ir.NameRef):
        target_type = symbols.check_type(stmt.target)
        if isinstance(target_type, np.dtype):
            target = symbols.make_alias(stmt.target)
            accountant.bind(target, value)
    return ir.InPlaceOp(value, stmt.pos)


@rewrite_stmt.register
def _(stmt: ir.SingleExpr, symbols: symbol_table, accountant: ValueLedger):
    value = accountant.rewrite(stmt.expr)
    return ir.SingleExpr(value, stmt.pos)


# branch_tester: uniformity_checker,
def rename_branch_values(branch, symbols: symbol_table, accountant: Optional[ValueLedger] = None):
    """
    start of if conversion

    Right now, this won't deal with nested loops (and may never, they're quite annoying)

    This will not deal with break or return either.

    The latter 2 are more tractable, but they can impact practical unrolling limits in a section, due to the need
    to check for all false predicates.

    :param if_branch:
    :param else_branch:
    :return:
    """

    # double check
    if contains_loops(branch):
        msg = f"Branch renaming cannot be used with nested loop constructs, pos: {branch[0].pos}"
        raise RuntimeError(msg)
    if accountant is None:
        # it feels like this should contain a joke about the cartesian product algorithm
        accountant = ValueLedger()
    reconstructed_branch = []

    for stmt in branch:
        if isinstance(stmt, ir.IfElse):
            # We don't use the concept of phi nodes here, since they aren't helpful at this level.
            # We just bind the original value name to a select statement, regardless of whether the branch
            # predicate is uniform.
            # Todo: This should clobber subscripted reads when passing subscripted writes.
            test = accountant.rewrite(stmt.test)
            if_expr_names = accountant.make_copy(copy_names=True, copy_exprs=True)
            reconstructed_if = rename_branch_values(stmt.if_branch, symbols, if_expr_names)
            # not sure if this can happen..
            if reconstructed_if is None:
                return
            else_expr_names = accountant.make_copy(copy_names=True, copy_exprs=True)
            reconstructed_else = rename_branch_values(stmt.else_branch, symbols, else_expr_names)
            if reconstructed_else is None:
                return
            if_stmts, if_accountant = reconstructed_if
            else_stmts, else_accountant = reconstructed_else

            # simple algorithm, basically ignores predicate uniformity at this point
            #
            # for each name, we check if it retrieves the same value. If so, that's the escaping value.
            # Otherwise we make a select statement, using the possibly uniform predicate, and bind the input name to it.
            # not identical to phi nodes, which don't help much here

            for name in if_accountant.names:
                if_value = if_accountant.current_value(name)
                else_value = else_accountant.current_value(name)
                if if_value == else_value:
                    # no select statement for any predicate
                    accountant.bind(name, if_value)
                else:
                    accountant.bind(name, ir.Select(test, if_value, else_value))
            # now we need to rebuild the statement
            rewrite = ir.IfElse(test, if_stmts, else_stmts, stmt.pos)
        elif isinstance(stmt, (ir.Assign, ir.InPlaceOp, ir.SingleExpr)):
            rewrite = rewrite_stmt(stmt, symbols, accountant)
        else:
            msg = f"Unexpected type {stmt}. If this comes up, it's almost certainly a bug."
            raise TypeError(msg)
        reconstructed_branch.append(rewrite)

    return reconstructed_branch, accountant


class DependencyGraph:
    # This approach is compatible with renaming, which is quite helpful.

    def __init__(self):
        self.cached = defaultdict(set)

    def __call__(self, expr):
        self.find_dependencies(expr)
        return self.cached[expr]

    @singledispatchmethod
    def find_dependencies(self, node):
        msg = f"No method to dependency check node type: {type(node)}, node: {node}."
        raise NotImplementedError(msg)

    @find_dependencies.register
    def _(self, node: ir.Constant):
        self.cached[node].add(node)
        return node

    @find_dependencies.register
    def _(self, node: ir.NameRef):
        self.cached[node].add(node)
        return node

    @find_dependencies.register
    def _(self, node: ir.Expression):
        deps = self.cached.get(node)
        if not deps:
            for subexpr in node.subexprs:
                subexpr_deps = self.find_dependencies(subexpr)
                self.cached[node].update(subexpr_deps)
            deps = self.cached[node]
        return deps
