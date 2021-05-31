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


class UpwardExposed(VisitorBase):
    """
    This embeds upward exposed information into predecessors of a given block.
    The information is calculated for entry and exit points of statement lists and ForLoop, IfElse, and WhileLoop
    IR constructs.

    This differs slightly from the same concept on a CFG in that statement lists may span multiple basic blocks.

    Suppose we have

    def f(x):
       i = 0
       for i in x:
          ...
       print(i)

    Here the for loop header's IR appears in the same statement list as the print statement, even though
    the first statement, the loop body, and the print statement would map to different basic blocks upon lowering.

    To avoid splitting the statement list or recording multiple offsets, we associate entry and exit info for
    any statement which marks an entry/exit point of a control flow region. This means that a statement list
    only records this info for the beginning and end of the list.

    Here the difference is that we don't associate entry/exit info for the for loop header with the enclosing statement
    list right before and right after. If we need that info, we retrieve it from the loop header instead.

    In a CFG format, the statements before and after the loop would be in different basic blocks, with their own
    local upward exposure information.

    """

    def __call__(self, entry):
        self.uevs = [set()]
        self.written = [set()]
        self.upward_exposed = {}
        self.visit(entry)

    @staticmethod
    def is_control_flow_entry_exit(node):
        return isinstance(node, (ir.IfElse, ir.ForLoop, ir.WhileLoop))

    def enter_scope(self):
        self.uevs.append(set())

    def exit_scope(self):
        scoped = self.uevs.pop()
        return scoped

    def register_write(self, name):
        self.uevs[-1].discard(name)

    @singledispatchmethod
    def register_read(self, node):
        # this could attempt to pretty print
        raise TypeError(f"No method known for read of type {type(node)}")

    @register_read.register
    def _(self, node: ir.Constant):
        pass

    @register_read.register
    def _(self, node: ir.NameRef):
        self.uevs[-1].add(node)

    @register_read.register
    def _(self, node: ir.Expression):
        for n in node.post_order_walk():
            self.register_read(n)

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: list):
        for stmt in reversed(list):
            self.visit(stmt)

    @visit.register
    def _(self, node: ir.ForLoop):
        self.visit(node.body)
        for t, v in node.walk_assignments():
            self.register_read(v)
            self.register_write(t)

    @visit.register
    def _(self, node: ir.Assign):
        self.visit(node.value)
        if isinstance(node.target, ir.Expression):
            self.register_read(node.target)
        else:
            # mark variable names that are written but not read
            if isinstance(node.value, ir.Expression):
                target = node.target
                if isinstance(target, ir.NameRef):
                    if all(target != subexpr for subexpr in node.value.post_order_walk()):
                        self.register_write(target)

    @visit.register
    def _(self, node: ir.IfElse):
        self.enter_scope()
        self.visit(node.if_branch)
        self.enter_scope()
        self.visit(node.else_branch)
        self.exit_scope()
        self.visit(node.test)

