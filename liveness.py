from contextlib import ContextDecorator, contextmanager
from functools import singledispatchmethod

import ir

from visitor import VisitorBase

"""
Need a map from id --> object

Need 

-------------------------------

This is easiest if we can avoid generating additional IR forms, which means attempting this with 
a tree based IR.


With that being noted, there are 3 things that might actually help here, noting that reaching checks precede this.

1. finding variables that may live across a loop latch (reaching check)
2. finding variables that may escape a given loop
3. finding variables that may escape a loop nest


This is a may reach problem.. so that doesn't work as well..



"""


class LoopClosure(ContextDecorator):

    def __enter__(self, visitor, header):
        self.visitor = visitor
        self.stashed = visitor.enclosing_loop
        visitor.enclosing_loop = header

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.visitor.enclosing_loop = self.stashed


class liveness_info:
    # branch points need an entry and exit
    def __init__(self, uevs):
        self.uevs = uevs


def is_control_flow_entry_exit(node):
    return isinstance(node, (ir.IfElse, ir.ForLoop, ir.WhileLoop))


class scoped:

    def __init__(self):
        self.uevs = {}
        self.gen = {}
        self.upward_exposed = None
        self.written = None
        self.key = None

    def register_scope(self, key):
        self.uevs[key] = set()
        self.gen[key] = set()

    def leave_scope(self):
        if self.key is not None:
            # may have a double entry if
            self.uevs[self.key] = self.upward_exposed
            self.gen[self.key] = self.gen
            self.upward_exposed = None
            self.written = None

    def change_scope(self, key):
        if self.key != key:
            self.leave_scope()
            self.register_scope(key)
            self.key = key
            self.upward_exposed = set()
            self.written = set()

    def register_read(self, target):
        if isinstance(target, ir.NameRef):
            if target not in self.written:
                self.upward_exposed.add(target)
        elif isinstance(target, ir.Expression):
            for subexpr in target.post_order_walk():
                if isinstance(subexpr, ir.NameRef):
                    if subexpr not in self.written:
                        self.upward_exposed.add(subexpr)

    def register_write(self, target):
        if isinstance(target, ir.NameRef):
            self.written.add(target)
        else:
            self.register_read(target)


class UpwardExposed(VisitorBase):

    def __init__(self):
        self.observer = None

    def __call__(self, entry):
        self.observer = scoped()
        self.visit(entry)
        uevs = self.observer.uevs
        gen = self.observer.gen
        self.handler = None
        return uevs, gen

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.IfElse):
        self.observer.register_read(node.test)
        self.visit(node.if_branch)
        self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.ForLoop):
        # header must have its own scope due to back edges
        self.observer.enter_scope(id(node))
        # This pass assumes we have already written out all simplifications
        # and are now dealing with a single loop index. Otherwise we have a bunch
        # of iterator assignments here that are only applied if entering the loop body.
        assert len(node.assigns) == 1
        target, iterable = node.assigns[0]
        self.observer.register_read(iterable)
        self.observer.register_write(target)
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.WhileLoop):
        self.observer.enter_scope(id(node))
        self.observer.register_read(node.test)
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.Assign):
        self.observer.register_read(node.value)
        self.observer.register_write(node.target)

    @visit.register
    def _(self, node: list):
        key = id(node[0]) if len(node) > 0 else id(node)
        self.observer.enter_scope(key)
        for stmt in node:
            self.visit(stmt)
        self.observer.leave_scope()


class LivenessSolver(VisitorBase):

    def __call__(self, entry, uevs, kills):
        self.uevs = uevs
        self.kills = kills
        self.uevs_local = None
        self.kills_local = None
        self.liveout = {}
        self.changed = True
        while self.changed:
            self.changed = False
            self.visit(entry)
        liveout = self.liveout
        self.uevs = self.read = self.written = self.kills = self.changed = None
        self.liveout = self.kills_local = self.uevs_local = self.liveout_local = None
        return liveout

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.ForLoop):
        while self.changed:
            self.changed = False
            self.visit(node.body)
        # visit header at end

    @visit.register
    def _(self, node: ir.IfElse):
        self.visit(node.if_branch)
        self.visit(node.else_branch)
        self.visit(node.test)

    @visit.register
    def _(self, node: list):
        # recompute liveness
        self.current_block = id(node)
        for index, stmt in enumerate(reversed(node)):
            self.visit(stmt)
            if is_control_flow_entry_exit(stmt):
                if index != len(node) - 1:
                    self.current_block = id(stmt)
