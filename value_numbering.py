import itertools

from functools import singledispatchmethod

import ir

from visitor import walk_branches


def number_subexpressions(node, labeled, gen):
    assert isinstance(node, ir.Expression)
    queued = [node]
    while queued:
        expr = queued.pop()
        if expr in labeled:
            continue


class ValueTracking:
    """
    Rethinking...
    We should be able to annotate each assignment with value numbers
    for parameters that have been set.

    This way we can check

    """

    def __init__(self):
        self.initial_values = {}
        self.exprs = {}
        self.subscript_writes = set()  # helps track write refs
        # Distinguish between last written and
        self.final_values = {}
        self.number_gen = itertools.count()

    def reset_assignments(self):
        # retain anything we have already numbered, just not the state of assignments
        self.final_values = {}
        self.subscript_writes = set()

    def lookup_name(self, name):
        value_id = self.final_values.get(name)
        if value_id is None:
            value_id = self.initial_values[name]
        return value_id

    def build_expr_id(self, expr):
        if isinstance(expr, ir.NameRef):
            key = self.lookup_name(expr)
        elif isinstance(expr, ir.Constant):
            key = expr
        elif isinstance(expr, ir.Expression):
            subexpr_nums = []
            for subexpr in expr.post_order_walk():
                if isinstance(subexpr, ir.Constant):
                    subexpr_nums.append(subexpr)
                elif isinstance(subexpr, ir.NameRef):
                    num = self.lookup_name(subexpr)
                    subexpr_nums.append(num)
            subexpr_nums = tuple(subexpr_nums)
            # Expressions are hashable, so pairing the expression itself
            # with the post ordered value numbers should be sufficient
            key = (expr, subexpr_nums)
            if key not in self.exprs:
                # record each unique construction once
                self.exprs[key] = next(self.number_gen)
        else:
            raise TypeError
        return key

    def lookup_value(self, ref):
        if isinstance(ref, ir.NameRef):
            value_id = self.lookup_name(ref)
        elif isinstance(ref, ir.Expression):
            value_id = self.build_expr_id(ref)
        elif isinstance(ref, ir.Constant):
            value_id = ref
        else:
            raise TypeError
        return value_id

    def register_references(self, node):
        if isinstance(node, ir.NameRef):
            if node not in self.initial_values:
                self.initial_values[node] = next(self.number_gen)
        elif isinstance(node, ir.Expression):
            for subexpr in node.post_order_walk():
                if isinstance(subexpr, ir.NameRef):
                    if subexpr not in self.initial_values:
                        self.initial_values = next(self.number_gen)

    def register_assignment(self, node: ir.Assign):
        target = node.target
        value = node.value
        value_id = self.lookup_value(value)
        if isinstance(target, ir.Subscript):
            target_id = self.lookup_value(node.target)
            self.subscript_writes.add(target_id)
        elif isinstance(target, ir.NameRef):
            self.final_values[target] = value_id


def contains_loops(node):
    return any(isinstance(stmt, (ir.ForLoop, ir.WhileLoop)) for stmt in walk_branches(node))


def contains_branches(node):
    return any(isinstance(stmt, (ir.IfElse, ir.CascadeIf)) for stmt in node)


def contains_control_flow(stmts):
    return any(isinstance(stmt, (ir.IfElse, ir.CascadeIf, ir.ForLoop, ir.WhileLoop)) for stmt in stmts)


def register_parameters(stmts, tracking):
    for stmt in stmts:
        if isinstance(stmt, ir.Assign):
            tracking.register_references(stmt.target)
            tracking.register_references(stmt.value)
        elif isinstance(stmt, ir.SingleExpr):
            tracking.register_references(stmt.expr)
        else:
            raise TypeError


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


def is_straightline_code(stmts):
    return all(isinstance(stmt, (ir.Assign, ir.SingleExpr)) for stmt in stmts)


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
            expr = stmt.expr
            if isinstance(expr, ir.Expression):
                add_subexprs(expr)


def number_branched_values(if_branch, else_branch):
    pass


def if_convert_branch(node: ir.IfElse, name_gen):
    """
    Branches that have varying conditions need to be if-converted in a way that
    """
    if not is_straightline_code(node.if_branch) or not is_straightline_code(node.else_branch):
        # not implemented here
        return
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
    pass

