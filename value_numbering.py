import itertools

from collections import defaultdict
from functools import singledispatchmethod

import ir

from visitor import walk_branches, VisitorBase


def is_control_flow_entry_exit(node):
    return isinstance(node, (ir.IfElse, ir.ForLoop, ir.WhileLoop))


def walk_assigns(stmts, reverse=False):
    if reverse:
        for stmt in reversed(stmts):
            if isinstance(stmt, ir.Assign):
                yield stmt.target, stmt.value
    else:
        for stmt in stmts:
            if isinstance(stmt, ir.Assign):
                yield stmt.target, stmt.value


def walk_expr_parameters(node: ir.Expression):
    for subexpr in node.post_order_walk():
        if not isinstance(subexpr, ir.Expression):
            yield subexpr


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

    @singledispatchmethod
    def register_read(self, target):
        raise NotImplementedError

    @register_read.register
    def _(self, target: ir.NameRef):
        if target not in self.gen[self.key]:
            self.uevs[self.key].add(target)

    @register_read.register
    def _(self, target: ir.Constant):
        self.constants.add(target)

    @register_read.register
    def _(self, target: ir.Expression):
        for subexpr in walk_expr_parameters(target):
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
        self.mem_writes = []  # track write ordering
        self.targets = set()
        # record value numbers that are bound to anything here
        self.values = set()
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

    @singledispatchmethod
    def register_assignment(self, target, value):
        raise NotImplementedError

    @register_assignment.register
    def _(self, target: ir.NameRef, value):
        value_id = self.get_or_create_ref(value)
        self.targets.add(target)
        self.last_value[target] = value_id

    @register_assignment.register
    def _(self, target: ir.Subscript, value):
        self.get_or_create_ref(value)
        target_id = self.get_or_create_ref(target)
        self.mem_writes.append(target_id)


def contains_loops(node):
    return any(isinstance(stmt, (ir.ForLoop, ir.WhileLoop)) for stmt in walk_branches(node))


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
    for target, value in walk_assigns(node):
        local_numbering.register_assignment(target, value)
    return local_numbering


def partition_by_mem_write(stmts):
    partitions = []
    current = []
    for stmt in stmts:
        current.append(stmt)
        if isinstance(stmt, ir.Assign):
            if isinstance(stmt.target, ir.Subscript):
                if current:
                    partitions.append(current)
                    current = []
    if current:
        partitions.append(current)
    return partitions


def partition_by_control_flow(stmts):
    partitions = []
    current = []
    for stmt in stmts:
        current.append(stmt)
        if isinstance(stmt, (ir.IfElse, ir.ForLoop, ir.WhileLoop)):
            if current:
                partitions.append(current)
                partitions.append([stmt])
                current = []
        else:
            current.append(stmt)
    if current:
        partitions.append(current)
    return partitions


def get_shared_target_names(node: ir.IfElse):
    """
    Find targets that may require ternary ops and/or renaming.

    """
    if any(is_control_flow_entry_exit(stmt) for stmt in itertools.chain(node.if_branch, node.else_branch)):
        raise ValueError
    if_targets = set()
    else_targets = set()
    for target, value in walk_assigns(node.if_branch):
        if isinstance(target, ir.NameRef):
            if_targets.add(target)
    for target, value in walk_assigns(node.else_branch):
        if isinstance(target, ir.NameRef):
            else_targets.add(target)
    shared = if_targets.intersection(else_targets)
    return shared


def get_final_assignments(stmts):
    assigns = {}
    for target, value in walk_assigns(stmts, reverse=True):
        if isinstance(target, ir.NameRef) and target not in assigns:
            assigns[target] = value
    return assigns


def predicate_branch(branch, local_if, local_else, combine_writes=False):
    # obvious stub
    return ()


def if_conversion(node: ir.IfElse, local_name_gen):

    if any(is_control_flow_entry_exit(stmt) for stmt in itertools.chain(node.if_branch, node.else_branch)):
        raise ValueError
    uevs, constants = UpwardExposed().find_upward_exposed(node)
    labeler = itertools.count()
    numbered_uevs = {}
    for v, i in zip(itertools.chain(uevs, constants), labeler):
        numbered_uevs[v] = i
    local_if = number_local_values(node.if_branch, numbered_uevs, numbered_uevs, labeler)
    local_else = number_local_values(node.else_branch, numbered_uevs, local_if.numbered, labeler)
    shared, reassigned = get_shared_target_names(node)
    repl = []
    test = ir.Assign(local_name_gen.make_name(), node.test, node.pos)
    repl.append(test)
    # Combine writes if write sequences match.
    # Otherwise we should order blocks and predicate writes.
    # Any values that appear on both sides can fall through to their uses
    # based on their first appearance.
    #
    # Subscripted writes should look like:
    #     original_array_name[value_numbered_index]
    combine_writes = local_if.mem_writes == local_else.mem_writes
    stmts = predicate_branch(node, local_if, local_else, combine_writes)
    return stmts
