import itertools

from collections import defaultdict
from functools import singledispatchmethod

import ir

from visitor import walk_branches, VisitorBase


def is_control_flow_entry_exit(node):
    return isinstance(node, (ir.IfElse, ir.ForLoop, ir.WhileLoop))


class scoped:

    def __init__(self):
        self.uevs = defaultdict(set)
        self.gen = defaultdict(set)
        self.constants = set()
        self.key = None

    def enter_scope(self, key):
        self.key = key

    def leave_scope(self):
        self.key = None

    def register_read(self, target):
        if isinstance(target, ir.NameRef):
            if target not in self.gen[self.key]:
                self.uevs[self.key].add(target)
        elif isinstance(target, ir.Constant):
            self.constants.add(target)
        elif isinstance(target, ir.Expression):
            for subexpr in target.post_order_walk():
                if not isinstance(subexpr, ir.Expression):
                    self.register_read(subexpr)

    def register_write(self, target):
        if isinstance(target, ir.NameRef):
            self.uevs[self.key].add(target)
        else:
            self.register_read(target)


class UpwardExposed(VisitorBase):
    """
    Simple tracking of upward exposed variables. This is less than what would be needed for full liveness
    analysis. It's intended to help with value numbering over acyclic control flow regions.

    """

    def __init__(self):
        self.observer = None

    def find_upward_exposed(self, entry):
        self.observer = scoped()
        self.visit(entry)
        uevs = self.observer.uevs
        constants = self.observer.constants
        self.observer = None
        return uevs, constants

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Constant):
        self.observer.register_read(node)

    @visit.register
    def _(self, node: ir.IfElse):
        self.visit(node.if_branch)
        self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.ForLoop):
        # Normal python semantics dictate that variables are bound
        # according to iterators if the loop body will be entered.
        # To preserve this behavior, assignments are treated as being performed
        # upon entering the loop body. Later on, we replace any header assignments
        # with a single induction variable that is scoped to the corresponding for loop
        # and therefore not allowed to escape.
        key = id(node.body)
        self.observer.enter_scope(key)
        for target, iterable in node.walk_assignments():
            self.observer.register_read(iterable)
            self.observer.register_write(target)
        self.visit(node.body)
        self.observer.leave_scope()

    @visit.register
    def _(self, node: ir.WhileLoop):
        # scope used for this header and nothing else
        # since it contains a test
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
        key = id(node[0])
        self.observer.enter_scope(key)
        for stmt in node:
            if key is None:
                # Some statement injected a new scope
                # We do it like this rather than as a
                # hierarchy to avoid certain issues.
                key = id(stmt)
                self.observer.enter_scope(key)
            self.visit(stmt)
        self.observer.leave_scope()


def enumerate_values(values, counter):
    numbered = {}
    for number, value in zip(counter, values):
        numbered[value] = number
    return numbered


class ValueTracking:
    """
    This tracks updates to various variable names in an acyclic context.
    It helps determine whether we require ternary or predicated ops across any
    particular boundary.

    Constants are explicitly tracked to avoid clashes between their hashes and the value
    numbering used.

    It's expected that upward exposed variables (uev) and constants are both mapped prior to this.
    In addition, number_gen must have a valid starting count so as to avoid conflicts.

    """

    def __init__(self, inputs, numbered, number_gen):
        # numbered tracks everything by value number
        self.numbered = numbered.copy()
        self.mem_ops = []  # track write ordering
        # record initial assignments as final
        # assignments on entry
        self.last_value = inputs.copy()
        self.number_gen = number_gen

    # returns true if this matches any upward exposed value

    def get_or_create_value_num(self, key):
        vn = self.last_value.get(key)
        if vn is None:
            vn = next(self.number_gen)
            # Bind an expression, based on value numbered
            # names and/or expressions with its own
            # unique value number.
            self.numbered[key] = vn
        return vn

    @singledispatchmethod
    def get_or_create_ref(self, tag):
        value_num = self.last_value.get(tag)
        assert value_num is not None
        return value_num

    @get_or_create_ref.register
    def _(self, tag: ir.BinOp):
        left = self.get_or_create_ref(tag.left)
        right = self.get_or_create_ref(tag.right)
        op = tag.op
        key = ir.BinOp(left, right, op)
        value_num = self.get_or_create_value_num(key)
        return value_num

    @get_or_create_ref.register
    def _(self, tag: ir.UnaryOp):
        operand = self.get_or_create_ref(tag.operand)
        op = tag.op
        key = ir.UnaryOp(operand, op)
        value_num = self.get_or_create_value_num(key)
        return value_num

    @get_or_create_ref.register
    def _(self, tag: ir.Subscript):
        value = self.get_or_create_value_num(tag.value)
        sl = self.get_or_create_value_num(tag.slice)
        value_num = ir.Subscript(value, sl)
        return value_num

    @get_or_create_ref.register
    def _(self, tag: ir.Slice):
        start = self.get_or_create_ref(tag.start)
        stop = self.get_or_create_ref(tag.stop)
        step = self.get_or_create_ref(tag.step)
        key = ir.Slice(start, stop, step)
        value_num = self.get_or_create_value_num(key)
        return value_num

    def register_assignment(self, node: ir.Assign):
        target = node.target
        value = node.value
        value_id = self.get_or_create_ref(value)
        if isinstance(value, ir.Subscript):
            self.mem_ops.append((value_id, "read"))
        if isinstance(target, ir.Subscript):
            target_id = self.get_or_create_ref(node.target)
            self.mem_ops.append((target_id, "write"))
        elif isinstance(target, ir.NameRef):
            # mark assignment to this variable name
            # with this value number
            self.last_value[target] = value_id
        else:
            # not particularly informative, since this uses an IR type.
            # it shouldn't come up outside of debugging contexts though.
            raise TypeError(f"Cannot register assign to type {type(target)}")


def contains_loops(node):
    return any(isinstance(stmt, (ir.ForLoop, ir.WhileLoop)) for stmt in walk_branches(node))


def get_required_tests(node: ir.IfElse):
    queued = [node.test]
    tests = set()
    while queued:
        test = queued.pop()
        if isinstance(test, ir.BinOp):
            tests.add(test)
        elif isinstance(test, ir.BoolOp):
            if test.op == "and":
                queued.extend(test.subexprs)
            else:
                # or isn't as easily optimizable
                return
    return tests


def get_expr_parameters(expr):
    if isinstance(expr, ir.Expression):
        return {name for name in expr.post_order_walk() if isinstance(name, ir.NameRef)}
    elif isinstance(expr, ir.NameRef):
        return {expr}
    else:
        return set()


def number_local_values(node: list, inputs, numbered, labeler):
    """
    node: statement list
    inputs:  {name: initial value number, ...}
    numbered: {key: value number}
    labeler: value number generator

    """
    for stmt in node:
        if is_control_flow_entry_exit(stmt):
            raise TypeError
    local_numbering = ValueTracking(inputs, numbered, labeler)
    for stmt in node:
        if isinstance(stmt, ir.Assign):
            pass
        elif isinstance(stmt, ir.SingleExpr):
            pass
        else:
            raise TypeError


def get_memory_writes(node: list):
    writes = []
    for stmt in node:
        if isinstance(stmt, ir.Assign):
            if isinstance(stmt.target, ir.Subscript):
                pass
                # writes.append()


def if_conversion(node: ir.IfElse):
    if any(is_control_flow_entry_exit(stmt) for stmt in itertools.chain(node.if_branch, node.else_branch)):
        raise ValueError
    uevs, constants = UpwardExposed().find_upward_exposed(node)
    labeler = itertools.count()
    # number uevs
    numbered_uevs = {}
    for v, i in zip(itertools.chain(uevs, constants), labeler):
        numbered_uevs[v] = i
    local_if = number_local_values(node.if_branch, numbered_uevs, numbered_uevs, labeler)
    # Retain existing expression numbers. Without this,
    # value numbering will alias across acyclic branch bounds.
    local_else = number_local_values(node.else_branch, numbered_uevs, local_if.numbered, labeler)
    min_params = None
    max_params = None
    test = node.test
    if isinstance(test, ir.BinOp):
        op = test.op
        if op in ("<", "<="):
            min_params = (test.left, test.right)
            max_params = (test.right, test.left)
        elif op in (">", ">="):
            min_params = (test.right, test.left)
            max_params = (test.left, test.right)
