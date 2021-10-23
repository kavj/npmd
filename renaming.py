import itertools

from contextlib import contextmanager
from collections import defaultdict
from functools import singledispatch, singledispatchmethod

import ir

from visitor import walk, StmtVisitor

# These yield everything including the original expression
# This way we can remove a lot of type checks from common paths.


def is_control_flow(node):
    return isinstance(node, (ir.IfElse, ir.ForLoop, ir.WhileLoop, ir.Break, ir.Continue))


def walk_params(expr):
    for subexpr in walk(expr):
        if isinstance(subexpr, ir.NameRef):
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
    def _(self, target: ir.ValueRef):
        for subexpr in walk_params(target):
            self.register_read(subexpr)

    def register_write(self, target):
        if isinstance(target, ir.NameRef):
            self.uevs[self.key].add(target)
        else:
            self.register_read(target)


@singledispatch
def find_upward_exposed(node):
    raise NotImplementedError


@find_upward_exposed.register
def _(node: ir.ForLoop):
    return find_upward_exposed(node.body)


@find_upward_exposed.register
def _(node: ir.WhileLoop):
    uevs = find_upward_exposed(node.body)
    test = node.test
    for subexpr in walk_params(test):
        uevs.add(subexpr)
    if isinstance(test, ir.NameRef):
        uevs.add(test)
    return uevs


@find_upward_exposed.register
def _(node: ir.IfElse):
    uevs = find_upward_exposed(node.if_branch)
    else_branch = find_upward_exposed(node.else_branch)
    uevs.update(else_branch)
    test = node.test
    for subexpr in walk_params(test):
        uevs.add(subexpr)
    if isinstance(test, ir.NameRef):
        uevs.add(test)


@find_upward_exposed.register
def _(node: list):
    uevs = set()
    gen = set()
    for stmt in node:
        if is_control_flow(stmt):
            break
        if isinstance(stmt, ir.Assign):
            target = stmt.target
            value = stmt.value
            if isinstance(value, ir.NameRef) and value not in gen:
                uevs.add(value)
            else:
                for param in walk_params(value):
                    uevs.add(param)
            if isinstance(stmt.target, ir.NameRef):
                gen.add(target)
            else:
                for param in walk_params(target):
                    uevs.add(param)
        else:
            # assume SingleExpr
            expr = stmt.expr
            for param in walk_params(expr):
                uevs.add(param)
    return uevs, gen


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
        self.by_value_num = []
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


class numbering_visitor(StmtVisitor):

    def __init__(self):
        self.inputs = None
        self.numbered = None
        self.labeler = None

    @contextmanager
    def numbering_context(self, inputs, numbered, labeler):
        self.inputs = inputs
        self.numbered = numbered
        self.labeler = labeler
        yield
        self.inputs = None
        self.numbered = None
        self.labeler = None

    def __call__(self, node, inputs, numbered, labeler):
        with self.numbering_context(inputs, numbered, labeler):
            self.visit(node)

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    # don't enter loops

    @visit.register
    def _(self, node: ir.ForLoop):
        pass

    @visit.register
    def _(self, node: ir.WhileLoop):
        pass

    @visit.register
    def _(self, node: ir.Assign):
        pass


def number_local_values(node: list, inputs, numbered, labeler):
    """
    node: statement list
    inputs:  {name: initial value number, ...}
    numbered: {key: value number}
    labeler: value number generator

    """
    for stmt in node:
        if is_control_flow(stmt):
            raise TypeError
    local_numbering = ValueTracking(inputs, numbered, labeler)
    # for target, value in walk_assigns(node):
    #    local_numbering.register_assignment(target, value)
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


# def get_shared_target_names(node: ir.IfElse):
#    """
#    Find targets that may require ternary ops and/or renaming.

#    """
#    if any(is_control_flow(stmt) for stmt in itertools.chain(node.if_branch, node.else_branch)):
#        raise ValueError
#    if_targets = set()
#    else_targets = set()
#    for target, value in walk_assigns(node.if_branch):
#        if isinstance(target, ir.NameRef):
#            if_targets.add(target)
#    for target, value in walk_assigns(node.else_branch):
#        if isinstance(target, ir.NameRef):
#            else_targets.add(target)
#    shared = if_targets.intersection(else_targets)
#    return shared


# def get_final_assignments(stmts):
#    assigns = {}
#    for target, value in walk_assigns(stmts, reverse=True):
#        if isinstance(target, ir.NameRef) and target not in assigns:
#            assigns[target] = value
#    return assigns


def predicate_branch(branch, local_if, local_else, combine_writes=False):
    # obvious stub
    return ()


def delay_assigns(names):
    """
    For each specified name, move all clobbering to the end of the block by making earlier
    assigns rely on single-use local variable names.
    """
    pass


def gather_read_written(block):
    """
    Record targets and values that are bound for a single block.
    """
    pass


def if_conversion(node: ir.IfElse, local_name_gen):
    """


    """

    if any(is_control_flow(stmt) for stmt in itertools.chain(node.if_branch, node.else_branch)):
        raise ValueError
    uevs, constants = find_upward_exposed(node)
    labeler = itertools.count()
    numbered_uevs = {}
    for v, i in zip(itertools.chain(uevs, constants), labeler):
        numbered_uevs[v] = i
    local_if = number_local_values(node.if_branch, numbered_uevs, numbered_uevs, labeler)
    local_else = number_local_values(node.else_branch, numbered_uevs, local_if.numbered, labeler)
    # shared, reassigned = get_shared_target_names(node)
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
