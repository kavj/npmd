from functools import cached_property, singledispatch, singledispatchmethod

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


class liveness_info:
    # branch points need an entry and exit
    def __init__(self, uevs):
        self.uevs = uevs


class UpwardExposed(VisitorBase):
    """

    """

    def __init__(self):
        self.read_first = None
        self.written = None

    def __call__(self, entry):
        self.uevs = {}
        self.kills = {}
        self.visit(entry)
        uevs = self.uevs
        kills = self.kills
        self.uevs = self.read = self.written = self.kills = None
        return uevs, kills

    def enter_scope(self, node):
        key = id(node)
        self.read_first = set()
        self.written = set()
        entry = self.uevs.get(key)
        # slightly weird, but this handles nodes that mark control flow points
        # embedded in a statement list
        if entry is None:
            self.uevs[key] = self.read_first
            self.kills[key] = self.written
        else:
            self.uevs[key] = (entry, self.read_first)
            self.kills[key] = (entry, self.written)

    def register_read(self, target):
        if isinstance(target, ir.NameRef):
            if target not in self.written:
                self.read_first.add(target)
        elif isinstance(target, ir.Expression):
            for subexpr in target.post_order_walk():
                if isinstance(subexpr, ir.NameRef):
                    if subexpr not in self.written:
                        self.read_first.add(subexpr)

    def register_write(self, target):
        if isinstance(target, ir.NameRef):
            self.written.add(target)
        else:
            self.register_read(target)

    @staticmethod
    def is_control_flow_entry_exit(node):
        return isinstance(node, (ir.IfElse, ir.ForLoop, ir.WhileLoop))

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.IfElse):
        self.enter_scope(node)
        self.register_read(node.test)
        self.visit(node.if_branch)
        self.visit(node.else_branch)
        # push an exit
        self.enter_scope(node)

    @visit.register
    def _(self, node: ir.ForLoop):
        # target and iterable list must not intersect
        self.enter_scope(node)
        targets = set()
        iterables = set()
        for t, i in node.walk_assignments():
            targets.add(t)
            iterables.add(i)
            self.register_read(i)
        assert not targets.intersection(iterables)
        self.visit(node.body)
        self.enter_scope(node)

    @visit.register
    def _(self, node: ir.WhileLoop):
        self.enter_scope(node)
        self.register_read(node.test)
        self.visit(node.body)
        self.enter_scope(node)

    @visit.register
    def _(self, node: ir.Assign):
        self.register_read(node.value)
        self.register_write(node.target)

    @visit.register
    def _(self, node: list):
        self.enter_scope(node)
        self.uevs[id(node)] = self.read_first
        for stmt in node:
            self.visit(stmt)
            if self.is_control_flow_entry_exit(stmt):
                self.enter_scope(stmt)
