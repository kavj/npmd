import itertools

from functools import singledispatchmethod

import ir

from visitor import walk_branches


class ValueTracking:
    def __init__(self):
        self.exprs = {}
        self.parameters = {}
        self.current = {}
        self.number_gen = itertools.count()

    def reset_assignments(self):
        self.current = {}

    @singledispatchmethod
    def get_value_number(self):
        raise TypeError

    @get_value_number.register
    def _(self, expr: ir.Expression):
        key = tuple(self.get_value_number(subexpr) for subexpr in expr.subexprs)
        value_num = self.exprs.get(key)
        if value_num is None:
            value_num = next(self.number_gen)
            self.exprs[key] = value_num
        return value_num

    @get_value_number.register
    def _(self, expr: ir.NameRef):
        value_num = self.current.get(expr)
        if value_num is None:
            value_num = self.parameters.get(expr)
            if value_num is None:
                raise ValueError
        return value_num

    @get_value_number.register
    def _(self, expr: ir.Constant):
        return self.parameters.get(expr)

    @singledispatchmethod
    def register_parameters(self, expr):
        raise NotImplementedError

    @register_parameters.register
    def _(self, expr: ir.Assign):
        self.register_parameters(expr.target)
        self.register_parameters(expr.value)

    @register_parameters.register
    def _(self, expr: ir.SingleExpr):
        self.register_parameters(expr.expr)

    @register_parameters.register
    def _(self, expr: ir.Expression):
        for subexpr in expr.post_order_walk():
            if isinstance(subexpr, ir.NameRef):
                if subexpr not in self.parameters:
                    self.parameters[subexpr] = next(self.number_gen)

    @register_parameters.register
    def _(self, expr: ir.NameRef):
        if expr not in self.parameters:
            self.parameters[expr] = next(self.number_gen)

    @register_parameters.register
    def _(self, expr: ir.Constant):
        if expr not in self.parameters:
            self.parameters[expr] = next(self.number_gen)

    def bind_value_to_name(self, target, value):
        if isinstance(target, ir.NameRef):
            value_num = self.get_value_number(value)
            self.current[target] = value_num


def contains_loops(node):
    return any(isinstance(stmt, (ir.ForLoop, ir.WhileLoop)) for stmt in walk_branches(node))


def contains_branches(node):
    return any(isinstance(stmt, (ir.IfElse, ir.CascadeIf)) for stmt in node)


def contains_control_flow(stmts):
    return any(isinstance(stmt, (ir.IfElse, ir.CascadeIf, ir.ForLoop, ir.WhileLoop)) for stmt in stmts)


def local_value_numbering(stmts, tracking=None):
    """
    stmts: statement sequence
    on_entry: definitions at entry
    components: variable names that parameterize a given expression, based on what is locally variant
    exprs: value numbering by expressions parameterization
    num_gen: number generator

    This is mostly used to factor common operations out of branch conditions for cases where they are

    """

    if tracking is None:
        tracking = ValueTracking()

    # register parameters appearing here
    for stmt in stmts:
        if not isinstance(stmt, (ir.Assign, ir.SingleExpr)):
            # should be partitioned block only
            raise TypeError
        tracking.register_parameters(stmt)

    for stmt in stmts:
        if isinstance(stmt, ir.Assign):
            tracking.bind_value_to_name(stmt.target, stmt.value)

    return tracking


def branch_value_numbering(node: ir.IfElse):
    if contains_control_flow(node.if_branch) or contains_control_flow(node.else_branch):
        # haven't decided yet
        raise ValueError
    tracking = ValueTracking()
    tracking.register_parameters(node.test)
    local_value_numbering(node.if_branch, tracking)
    assigned_in_if = tracking.current
    tracking.reset_assignments()
    local_value_numbering(node.else_branch, tracking)
    assigned_in_else = tracking.current
    return tracking, assigned_in_if, assigned_in_else


def block_partition(stmts):
    partitions = []
    curr = []
    for stmt in stmts:
        if isinstance(stmt, (ir.IfElse, ir.CascadeIf, ir.ForLoop, ir.WhileLoop)):
            if curr:
                partitions.append(curr)
                curr = []
            partitions.append(stmt)
    if curr:
        partitions.append(curr)
    return tuple(partitions)
