import itertools

from functools import singledispatchmethod

import ir

from visitor import walk_branches, VisitorBase


def number_subexpressions(node, labeled, gen):
    assert isinstance(node, ir.Expression)
    queued = [node]
    while queued:
        expr = queued.pop()
        if expr in labeled:
            continue


def is_control_flow_entry_exit(node):
    return isinstance(node, (ir.IfElse, ir.ForLoop, ir.WhileLoop))


class scoped:

    def __init__(self, uevs, gen):
        self.uevs = uevs
        self.gen = gen
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
    """
    Simple tracking of upward exposed variables. This is less than what would be needed for full liveness
    analysis. It's intended to help with value numbering over acyclic control flow regions.

    """

    def __init__(self):
        self.observer = None

    def __call__(self, entry):
        uevs = {}
        gen = {}
        self.observer = scoped(uevs, gen)
        self.visit(entry)
        self.scope = None
        self.observer = None
        return uevs, gen

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

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

    """

    def __init__(self, uev):
        self.initial_values = uev
        self.exprs = {}
        self.subscript_writes = set()  # helps track write refs
        # Distinguish between last written and
        self.assigned_values = {}
        self.number_gen = itertools.count()

    def reset_assignments(self):
        # retain anything we have already numbered, just not the state of assignments
        self.assigned_values = {}
        self.subscript_writes = set()

    def _is_initial_value(self, tag):
        """
        Check whether a name or expression is generated prior to any binding operation
        that affects it.

        This treats any reaching of the initial value as True
        For example

        c = a
        a = b
        a = c

        would return True for "a" after the last assignment, since we are dealing with trivial
        copy assignments, and non-aliasing named variables.

        """

        if isinstance(tag, ir.Expression):
            for subexpr in tag.post_order_walk():
                if not isinstance(subexpr, ir.Expression):
                    if subexpr in self.assigned_values:
                        return False
        elif isinstance(tag, ir.Constant):
            return True
        else:
            return tag not in self.assigned_values

    def _get_or_create_value_num(self, key):
        value_num = self.assigned_values.get(key)
        if value_num is None:
            value_num = self.initial_values.get(key)
            if value_num is None:
                value_num = next(self.number_gen)
                if self._is_initial_value(key):
                    self.initial_values[key] = value_num
        return value_num

    @singledispatchmethod
    def get_or_create_ref(self, tag):
        value_num = self.assigned_values.get(tag)
        if value_num is None:
            value_num = self.initial_values.get(tag)
            if value_num is None:
                value_num = next(self.number_gen)
                if self._is_initial_value(tag):
                    self.initial_values[tag] = value_num
        return value_num

    @get_or_create_ref.register
    def _(self, tag: ir.BinOp):
        left = self.get_or_create_ref(tag.left)
        right = self.get_or_create_ref(tag.right)
        op = tag.op
        key = ir.BinOp(left, right, op)
        value_num = self._get_or_create_value_num(key)
        return value_num

    @get_or_create_ref.register
    def _(self, tag: ir.UnaryOp):
        operand = self.get_or_create_ref(tag.operand)
        op = tag.op
        key = ir.UnaryOp(operand, op)
        value_num = self._get_or_create_value_num(key)
        return value_num

    @get_or_create_ref.register
    def _(self, tag: ir.Slice):
        start = self.get_or_create_ref(tag.start)
        stop = self.get_or_create_ref(tag.stop)
        step = self.get_or_create_ref(tag.step)
        key = ir.Slice(start, stop, step)
        value_num = self._get_or_create_value_num(key)
        return value_num

    def register_assignment(self, node: ir.Assign):
        target = node.target
        value = node.value
        value_id = self.get_or_create_ref(value)
        if isinstance(target, ir.Subscript):
            target_id = self.get_or_create_ref(node.target)
            self.subscript_writes.add(target_id)
        elif isinstance(target, ir.NameRef):
            self.assigned_values[target] = value_id


def contains_loops(node):
    return any(isinstance(stmt, (ir.ForLoop, ir.WhileLoop)) for stmt in walk_branches(node))


def contains_control_flow(stmts):
    return any(isinstance(stmt, (ir.IfElse, ir.CascadeIf, ir.ForLoop, ir.WhileLoop)) for stmt in stmts)


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


def gather_referenced(stmts):
    refs = set()

    def add_subexprs(expr):
        for subexpr in expr.post_order_walk():
            if isinstance(subexpr, ir.NameRef):
                refs.add(subexpr)

    for stmt in stmts:
        if isinstance(stmt, ir.Assign):
            target = stmt.target
            if isinstance(target, ir.Expression):
                add_subexprs(target)
            value = stmt.value
            if isinstance(value, ir.Expression):
                add_subexprs(value)
            elif isinstance(value, ir.NameRef):
                refs.add(value)
        else:  # single expression
            if isinstance(stmt.expr, ir.Expression):
                add_subexprs(stmt.expr)


def number_branched_values(if_branch, else_branch):
    pass


def if_convert_branch(node: ir.IfElse, name_gen):
    """
    Branches that have varying conditions need to be if-converted in a way that
    """

    # Check whether we may be able to convert to min/max
    test = node.test
    min_params = None
    max_params = None
    if isinstance(test, ir.BinOp):
        op = test.op
        if op in ("<", "<="):
            min_params = (test.left, test.right)
            max_params = (test.right, test.left)
        elif op in (">", ">="):
            min_params = (test.right, test.left)
            max_params = (test.left, test.right)
