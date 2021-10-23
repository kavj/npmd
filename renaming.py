import itertools

from contextlib import contextmanager
from collections import Counter, defaultdict, namedtuple
from functools import singledispatch, singledispatchmethod

import ir

from utils import unpack_iterated
from visitor import walk, StmtVisitor

# These yield everything including the original expression
# This way we can remove a lot of type checks from common paths.


def is_control_flow(node):
    return isinstance(node, (ir.IfElse, ir.ForLoop, ir.WhileLoop, ir.Break, ir.Continue))


def walk_params(expr):
    for subexpr in walk(expr):
        if isinstance(subexpr, ir.NameRef):
            yield subexpr


liveness_info = namedtuple('liveness_info', 'uev read, writes')


class LivenessDetail(StmtVisitor):
    """
    Kills and upward exposed variable information can be used for liveness analysis.
    Right now we're primarily using them for local renaming.

    Writes track the number of times something is written, since this is one factor in determining
    whether something should be renamed within straight line code.

    """

    def __init__(self):
        self.entry_point = None
        self.li = None

    @contextmanager
    def liveness_context(self, entry_point):
        self.li = liveness_info(uev=set(), read=set(), writes=Counter())
        yield
        assert self.entry_point is entry_point
        self.li = None

    def __call__(self, node):
        with self.liveness_context(node):
            self.visit(node)
            li = self.li
        return li

    def register_reads(self, node):
        if isinstance(node, ir.NameRef):
            self.li.read.add(node)
            if node not in self.killed:
                self.li.uev.add(node)
        else:
            for compon in walk_params(node):
                self.li.read.add(compon)
                if compon not in self.killed:
                    self.li.uev.add(compon)

    def register_writes(self, node):
        if isinstance(node, ir.NameRef):
            self.li.writes[node] += 1
        else:
            self.register_reads(node)

    @singledispatchmethod
    def visit(self, node):
        assert self.entry_point is None
        return super().visit(node)

    @visit.register
    def _(self, node: ir.Assign):
        self.register_reads(node.value)
        self.register_writes(node.target)

    @visit.register
    def _(self, node: ir.Function):
        for arg in node.args:
            self.register_writes(arg)
        self.visit(node.body)

    @visit.register
    def _(self, node: list):
        for stmt in node:
            if isinstance(stmt, (ir.ForLoop, ir.WhileLoop, ir.IfElse)):
                return
            self.visit(stmt)

    @visit.register
    def _(self, node: ir.ForLoop):
        for target, value in unpack_iterated(node):
            self.register_reads(value)
            self.register_writes(target)

    @visit.register
    def _(self, node: ir.WhileLoop):
        self.register_reads(node.test)

    @visit.register
    def _(self, node: ir.IfElse):
        self.register_reads(node.test)

    @visit.register
    def _(self, node: ir.Function):
        for arg in node.args:
            self.register_writes(arg)


class Renamer(StmtTransformer):

    """
    Local renaming
    """

    def __init__(self, ctx):



class LVN(StmtTransformer):
    """
    Local value numbering, with some utility for if branch conversion
    """

    def __init__(self, syms, types):
        self.syms = syms
        self.types = types
        self.collect_targets = TargetCollector()
        self.labeler = None

    def __call__(self, node):
        if not isinstance(node, (ir.Function, ir.ForLoop, ir.WhileLoop)):
            raise TypeError(f"Not a valid entry point: {node}")
        with self.vn_barrier(node.body, node.pos):
            self.visit(node.body)

    def copy_out(self, append_to, pos):
        assert self.labeler is not None
        for target, value in self.labeler.source_assigns:
            assign = ir.Assign(target, value, pos)
            append_to.append(assign)

    def clobber_vns(self, append_to, pos):
        stashed = self.labeler
        if stashed is not None:
            self.copy_out(append_to, pos)
        self.labeler = Labeler(self.syms, self.types)
        yield
        self.copy_out(append_to, pos)
        self.labeler = stashed

    @contextmanager
    def vn_barrier(self, append_to, pos):
        """
        Clobbering boundary for value numbering
        """
        have_existing = self.labeler is not None
        if have_existing:
            self.copy_out(append_to, pos)
        self.labeler = Labeler(self.syms, self.types)
        yield
        self.copy_out(append_to, pos)
        if have_existing:
            self.labeler = Labeler(self.syms, self.types)
        else:
            self.labeler = None

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: list):
        repl = []
        for stmt in node:
            stmt = self.visit(stmt)
            if isinstance(stmt, (ir.ForLoop, ir.WhileLoop)):
                with self.vn_barrier(repl, stmt.pos):
                    self.visit(stmt.body)
            elif isinstance(stmt, ir.IfElse):

                self.clobber_vns(repl, stmt.pos)
        return repl

    @visit.register
    def _(self, node: ir.Assign):
        if isinstance(node.target, ir.NameRef):
            self.labeler.gen_assign(node.target, node.value)
        return node

    @visit.register
    def _(self, node: ir.IfElse):
        if node.if_branch and node.else_branch:
            # inherit existing assignments but
            assigned = self.labeler.assigned.copy()
            self.visit(node.if_branch)
            if_assigns = self.labeler.assigned
            self.labeler.clear_assigns()
            self.labeler.assigned = assigned
            self.visit(node.else_branch)
            else_assigns = self.labeler.assigned
            # check for clobbered in sub-branches
            clobbered_if = self.collect_targets(node.if_branch)
            clobbered_else = self.collect_targets(node.else_branch)
            must_clobber = clobbered_if.union(clobbered_else)
            seen_both = set(if_assigns.keys()).intersection(else_assigns.keys())
            safe = set()
            for key in seen_both:
                on_if = if_assigns.get(key)
                on_else = else_assigns.get(key)
                if on_if is not None and on_if == on_else:
                    safe.add(key)
            for s in safe:
                must_clobber.pop(s)
            # need to clear up assignments here
        else:
            # only one non-trivial, any assigns must be marked clobbered
            clobbered_if = self.collect_targets(node.if_branch)
            clobbered_else = self.collect_targets(node.else_branch)
            must_clobber = clobbered_if.union(clobbered_else)
            self.visit(node.if_branch)
            self.visit(node.else_branch)
        self.labeler.clear_assigns(must_clobber)

    @visit.register
    def _(self, node: ir.ForLoop):
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.WhileLoop):
        self.visit(node.body)
