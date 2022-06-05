import itertools

from collections import defaultdict
from functools import singledispatch, singledispatchmethod
from typing import Dict, Iterable, List, Set, Union
from weakref import WeakValueDictionary

import ir

from analysis import find_ephemeral_references
from symbol_table import SymbolTable
from utils import is_entry_point
from traversal import sequence_block_intervals, walk_parameters, StmtTransformer


class ExpandAssigns(StmtTransformer):

    def __init__(self, symbols: SymbolTable):
        self.symbols = symbols

    def visit(self, node):
        if isinstance(node, ir.InPlaceOp):
            if (isinstance(node.target, ir.NameRef)
                    and not isinstance(self.symbols.check_type(node.target), ir.ArrayType)):
                # we already record target on these, since any unordered binop
                # types use lexical reordering so that (where op is unordered),
                # 'a op b' hashes equivalent to 'b op a'. Because of this,
                # we have to explicitly record a target on inplace ops.
                node = ir.Assign(node.target, node.value, node.pos)
            return node
        else:
            return super().visit(node)


def get_augmented_name(stmt: Union[ir.Assign, ir.InPlaceOp]):
    """
    Returns a target if any that is augmented rather than bound by this operation.
    :param stmt:
    :return:
    """

    assert isinstance(stmt, (ir.Assign, ir.InPlaceOp))
    if isinstance(stmt.target, ir.Subscript):
        return stmt.target.value
    elif isinstance(stmt, ir.InPlaceOp):
        return stmt.target


def make_use_def_map(block: Iterable[ir.StmtBase]):
    """
    maps each statement to the statements it directly depends on
    within a single control flow block (no back edges or branches)
    :param block:
    :return:
    """

    def update_expr_dependencies(expr: ir.ValueRef, stmt: ir.StmtBase):
        nonlocal depends_on
        for name in walk_parameters(expr):
            most_recent_assign_to = name_to_latest_assign.get(name)
            if most_recent_assign_to is not None:
                depends_on[stmt].add(most_recent_assign_to)

    def update_linked(stmt: Union[ir.Assign, ir.InPlaceOp], target_name: ir.NameRef):
        # linked indicates we have an assignment which is only known to augment some other value
        # this avoids an augmenting assignment to some array element being the only thing to keep
        # an array binding op alive
        nonlocal linked
        nonlocal depends_on
        assert isinstance(target_name, ir.NameRef)
        existing_assign = name_to_latest_assign.get(target_name)
        if existing_assign is not None:
            linked[existing_assign].add(stmt)
        for name in itertools.chain(walk_parameters(stmt.value), walk_parameters(stmt.target)):
            if name != target_name:
                most_recent_assign_to = name_to_latest_assign.get(name)
                if most_recent_assign_to is not None:
                    depends_on[stmt].add(most_recent_assign_to)

    name_to_latest_assign = WeakValueDictionary()  # only binding operations
    linked = defaultdict(set)
    depends_on = defaultdict(set)
    for stmt in block:
        if isinstance(stmt, (ir.Assign, ir.InPlaceOp)):
            augmented_name = get_augmented_name(stmt)
            if augmented_name is not None:
                update_linked(stmt, augmented_name)
            else:
                target = stmt.target
                value = stmt.value
                update_expr_dependencies(value, stmt)
                if isinstance(stmt, ir.Assign):
                    if isinstance(target, ir.NameRef):
                        # binding operation
                        name_to_latest_assign[target] = stmt
                    else:
                        # Specific to Assign, since InplaceOp gets this by traversing value
                        update_expr_dependencies(target, stmt)
                else:
                    # this means things of the form {a, a[...]} += ...  depend on array 'a' being bound somewhere
                    update_expr_dependencies(target, stmt)

        elif isinstance(stmt, (ir.SingleExpr, ir.Return)):
            # We don't have to care about any branching statements, since they will always reference
            # values that are not ephemeral.
            if stmt.value is not None:
                # can only be None for return
                update_expr_dependencies(stmt.value, stmt)
    return depends_on, name_to_latest_assign, linked


def update_liveness(stmt: ir.StmtBase,
                    depends_on: Dict[ir.StmtBase, Set[ir.StmtBase]],
                    linked: Dict[ir.StmtBase, Set[ir.StmtBase]],
                    live_stmts: Set[ir.StmtBase]):
    """
    marks stmt, any linked statements, and each of their dependencies as live

    Since linked statements follow an initial binding statement, it's possible
    to otherwise disover that they are live after passing.

    :param stmt:
    :param depends_on:
    :param linked:
    :param live_stmts:
    :return:
    """
    live_stmts.add(stmt)
    deps = depends_on.get(stmt)
    queued = [deps] if deps else []
    linked_stmts = linked.get(stmt)
    if linked_stmts:
        queued.append(linked_stmts)
    while queued:
        stmts = queued.pop()
        for s in stmts:
            if s not in live_stmts:
                live_stmts.add(s)
                linked_stmts = linked.get(s)
                if linked_stmts:
                    queued.append(linked_stmts)
                deps = depends_on.get(s)
                if deps:
                    queued.append(deps)


def dead_code_elim(block: List[ir.StmtBase], ephemeral_refs: Set[ir.NameRef]):
    depends_on, name_to_latest, linked_stmts = make_use_def_map(block)
    live_stmts = set()
    for stmt in reversed(block):
        if stmt in live_stmts:
            continue  # already handled by something that declared it a dependency
        if isinstance(stmt, (ir.Assign, ir.InPlaceOp)):
            augmented = get_augmented_name(stmt)
            if augmented is None:
                # binding op
                name: ir.NameRef = stmt.target
                if stmt not in live_stmts:
                    if name not in ephemeral_refs and stmt == name_to_latest[name]:
                        # this is the only assignment whose resulting value can escape
                        # some non-ephemeral name. This doesn't mean it will, but we aren't checking that.
                        update_liveness(stmt, depends_on, linked_stmts, live_stmts)
        elif isinstance(stmt, (ir.SingleExpr, ir.Return)):
            update_liveness(stmt, depends_on, linked_stmts, live_stmts)
        else:
            live_stmts.add(stmt)
    # Now eliminate any statements that aren't referenced
    repl = [stmt for stmt in block if stmt in live_stmts]
    return repl


class LVNRewriter:

    def __init__(self, should_rename: Set[ir.ValueRef], symbols: SymbolTable):
        self.should_rename = should_rename
        self.cached = {}
        self.symbols = symbols
        self.expr_to_lvn = {}
        # we want to maintain both should rename and name to latest
        # without both, it's more error prone to determine what has been bound
        self.name_to_latest = {}

    @classmethod
    def rewrite_block(cls, block: List[ir.StmtBase], rename_targets: Set[ir.NameRef], symbols: SymbolTable):
        rewriter = cls(rename_targets, symbols)
        repl = []
        for stmt in block:
            repl.append(rewriter.rewrite_statement(stmt))
        return repl

    def register_write(self, target: ir.NameRef, value: ir.ValueRef):
        if target in self.should_rename:
            name = self.symbols.make_versioned(target)
            self.name_to_latest[target] = name
            self.expr_to_lvn[value] = name
            target = name
        return target

    @singledispatchmethod
    def rewrite(self, expr):
        msg = f'No method to rewrite "{expr}".'
        raise TypeError(msg)

    @rewrite.register
    def _(self, expr: ir.Expression):
        if expr in self.cached:
            return expr
        repl = expr.reconstruct(*(self.rewrite(subexpr) for subexpr in expr.subexprs))
        repl = self.expr_to_lvn.get(repl, repl)
        if repl == expr:
            self.cached[expr] = expr
            return expr
        else:
            self.cached[expr] = repl
            return repl

    @rewrite.register
    def _(self, expr: ir.NameRef):
        return self.name_to_latest.get(expr, expr)

    @rewrite.register
    def _(self, expr: ir.ValueRef):
        return expr

    @singledispatchmethod
    def rewrite_statement(self, stmt):
        msg = f"No rewriter for {stmt}"
        raise TypeError(msg)

    @rewrite_statement.register
    def _(self, stmt: ir.StmtBase):
        return stmt

    @rewrite_statement.register
    def _(self, stmt: ir.Assign):
        target = stmt.target
        value = self.rewrite(stmt.value)
        if isinstance(target, ir.NameRef) and target in self.should_rename:
            target = self.register_write(target, value)
        pos = stmt.pos
        repl = ir.Assign(target, value, pos)
        if repl == stmt:
            return stmt
        return repl

    @rewrite_statement.register
    def _(self, stmt: ir.InPlaceOp):
        target = self.rewrite(stmt.target)
        value = self.rewrite(stmt.value)
        pos = stmt.pos
        repl = ir.InPlaceOp(target, value, pos)
        if repl == stmt:
            return stmt
        return repl

    @rewrite_statement.register
    def _(self, stmt: ir.SingleExpr):
        value = self.rewrite(stmt.value)
        pos = stmt.pos
        repl = ir.SingleExpr(value, pos)
        if repl == stmt:
            return stmt
        return repl


def get_single_block_rename_targets(block: Iterable[ir.StmtBase], ephemeral_refs: Set[ir.NameRef]):
    """
    This checks for a pattern of
    two groups of reads, separated by one or more writes.

    :param block:
    :param ephemeral_refs:
    :return:
    """
    war = set()
    read = set()
    must_rename = set()

    def update_read(v: ir.ValueRef):
        for p in walk_parameters(v):
            if p in war:
                must_rename.add(p)
            else:
                read.add(p)

    for stmt in block:

        if isinstance(stmt, ir.Assign):
            update_read(stmt.value)
            target = stmt.target
            if isinstance(target, ir.NameRef):
                if target in read:
                    war.add(target)
        elif isinstance(stmt, ir.InPlaceOp):
            target = stmt.target
            update_read(stmt.value)
            if isinstance(target, ir.NameRef):
                # If this is ephemeral and not predicated, then we can convert
                # this to out of place in renaming
                if target in ephemeral_refs:
                    if target in war:
                        must_rename.add(target)
                    else:
                        war.add(target)
            else:
                update_read(target)
        elif isinstance(stmt, ir.SingleExpr) or (isinstance(stmt, ir.Return) and stmt.value is not None):
            update_read(stmt.value)

    return must_rename


def optimize_block(block: List[ir.StmtBase], symbols: SymbolTable, ephemeral_refs: Set[ir.NameRef]):
    block = list(block)
    block = dead_code_elim(block, ephemeral_refs)
    should_rename = get_single_block_rename_targets(block, ephemeral_refs)
    block = LVNRewriter.rewrite_block(block, should_rename, symbols)
    return block


@singledispatch
def run_local_opts(node, symbols: SymbolTable, ephemeral_refs: Set[ir.NameRef]):
    msg = f'"{node}" does not have a valid type for optimize region.'
    raise TypeError(msg)


@run_local_opts.register
def _(node: list, symbols: SymbolTable, ephemeral_refs: Set[ir.NameRef]):
    # check if this is a single block
    repl = []
    for start, stop in sequence_block_intervals(node):
        assert 0 <= start < stop
        if is_entry_point(node[start]):
            stmt = run_local_opts(node[start], symbols, ephemeral_refs)
            repl.append(stmt)
        else:
            stmts = optimize_block(node[start:stop], symbols, ephemeral_refs)
            repl.extend(stmts)
    return repl


@run_local_opts.register
def _(node: ir.IfElse, symbols: SymbolTable, ephemeral_refs: Set[ir.NameRef]):
    test = node.test
    pos = node.pos

    if node.if_branch:
        if_branch = run_local_opts(node.if_branch, symbols, ephemeral_refs)
    else:
        if_branch = []

    if node.else_branch:
        else_branch = run_local_opts(node.else_branch, symbols, ephemeral_refs)
    else:
        else_branch = []

    return ir.IfElse(test, if_branch, else_branch, pos)


@run_local_opts.register
def _(node: ir.ForLoop, symbols: SymbolTable, ephemeral_refs: Set[ir.NameRef]):
    target = node.target
    iterable = node.iterable
    body = run_local_opts(node.body, symbols, ephemeral_refs)
    pos = node.pos
    return ir.ForLoop(target, iterable, body, pos)


@run_local_opts.register
def _(node: ir.WhileLoop, symbols: SymbolTable, ephemeral_refs: Set[ir.NameRef]):
    test = node.test
    body = run_local_opts(node.body, symbols, ephemeral_refs)
    pos = node.pos
    return ir.WhileLoop(test, body, pos)


def run_func_local_opts(node: ir.Function, symbols: SymbolTable):
    # do this before checking ephemeral references, since it may reveal more
    node = ExpandAssigns(symbols).visit(node)
    ephemeral_refs = find_ephemeral_references(node)
    # have to handle sequencing manually here for context
    body = run_local_opts(node.body, symbols, ephemeral_refs)
    return ir.Function(node.name, node.args, body)
