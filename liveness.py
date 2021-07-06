from functools import singledispatchmethod

import ir

from visitor import VisitorBase

"""
This is an extremely simple pass, to determine whether an initial definition for some variable precedes
its scope. It could merge with value numbering. The purpose is to determine what variables retrieve an initial
value from some enclosing or preceding scope. 

"""


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
            self.key = None

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
        self.observer = None
        return uevs, gen

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.IfElse):
        self.observer.register_read(node.test)
        self.visit(node.if_branch)
        self.visit(node.else_branch)
        self.observer.leave_scope()

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
        self.observer.leave_scope()

    @visit.register
    def _(self, node: ir.WhileLoop):
        self.observer.enter_scope(id(node))
        self.observer.register_read(node.test)
        self.visit(node.body)
        self.observer.leave_scope()

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
