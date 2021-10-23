import itertools

from contextlib import contextmanager
from collections import defaultdict, namedtuple
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


liveness_info = namedtuple('liveness_info', 'uev read, killed')

class LivenessDetail(StmtVisitor):
    """
    Kills and upward exposed variable information can be used for liveness analysis.
    Right now we're primarily using them for local renaming.

    """

    def __init__(self):
        self.entry_point = None
        self.li = None

    @contextmanager
    def liveness_context(self, entry_point):
        self.li = liveness_info(uev=set(), read=set(), killed=set())
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
            self.li.killed.add(node)
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
        have_existing = self.labeler is not None
        if self.labeler is not None:
            self.copy_out(append_to, pos)
        self.labeler = Labeler(self.syms, self.types)
        yield
        self.copy_out(append_to, pos)
        if have_existing:
            self.labeler = Labeler(self.syms, self.types)
        else:
            self.labeler = None

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
