from functools import singledispatchmethod, lru_cache
from weakref import WeakKeyDictionary

import ir


def walk_assigns(stmts, reverse=False):
    if reverse:
        for stmt in reversed(stmts):
            if isinstance(stmt, ir.Assign):
                yield stmt.target, stmt.value
    else:
        for stmt in stmts:
            if isinstance(stmt, ir.Assign):
                yield stmt.target, stmt.value


def walk_expr_parameters(node):
    """
    Walk an expression, yielding only sub-expressions that are not expressions themselves.

    """
    if hasattr(node, "subexprs"):
        for subexpr in node.subexprs:
            if hasattr(subexpr, "subexprs"):
                yield from walk_expr_parameters(subexpr)
            else:
                yield subexpr


def walk_expr(node):
    """
    walk an expression depth first in post order, yielding everything but the original node
    """
    if hasattr(node, "subexprs"):
        for subexpr in node.subexprs:
            yield from walk_expr(subexpr)
            yield subexpr


def walk_statements(node):
    """
    extending walk interface to include lists
    """

    if isinstance(node, list):
        for stmt in node:
            yield stmt
            if isinstance(stmt, (ir.ForLoop, ir.WhileLoop, ir.IfElse)):
                yield from walk_statements(stmt)
    elif isinstance(node, (ir.ForLoop, ir.WhileLoop)):
        yield from walk_statements(node.body)
    elif isinstance(node, ir.IfElse):
        yield from walk_statements(node.if_branch)
        yield from walk_statements(node.else_branch)
    else:
        raise TypeError(f"Cannot walk type of {type(node)}.")


def walk_branches(node):
    """
    Walk statements, expanding branches but not loop constructs

    """
    if isinstance(node, ir.IfElse):
        yield from walk_branches(node.if_branch)
        yield from walk_branches(node.else_branch)
    elif isinstance(node, list):
        for stmt in node:
            yield stmt
            if isinstance(stmt, ir.IfElse):
                yield from walk_branches(stmt.if_branch)
                yield from walk_branches(stmt.else_branch)


class ExpressionVisitor:
    """
    Since expressions are immutable and hashable, they deserve their own visitor.
    This

    """

    def __init__(self):
        self.rules = WeakKeyDictionary()

    def clear(self):
        self.rules.clear()
        self.lookup.cache_clear()

    def invalidate_cache(self):
        self.lookup.cache_clear()

    def invalidate_rule(self, expr):
        if expr in self.rules:
            self.rules.pop(expr)
            self.lookup.cache_clear()

    def assign_rule(self, expr, output):
        self.rules[expr] = output
        self.lookup.cache_clear()  # invalidate cache

    @lru_cache
    def lookup(self, expr):
        output = self.rules.get(expr)
        if output is None:
            # try rewriting
            output = self._rewrite(expr)
        # avoid propagating copies
        return output if output != expr else expr

    @singledispatchmethod
    def _rewrite(self, expr):
        msg = f"No method to rewrite object of type {type(expr)}."
        raise NotImplementedError(msg)

    @_rewrite.register
    def _(self, expr: ir.Length):
        value = self.lookup(expr.value)
        return ir.Length(value)

    @_rewrite.register
    def _(self, expr: ir.Subscript):
        value = self.lookup(expr.value)
        slice_ = self.lookup(expr.slice)
        return ir.Subscript(value, slice_)

    @_rewrite.register
    def _(self, expr: ir.Min):
        values = tuple(self.lookup(value) for value in expr.values)
        return ir.Min(values)

    @_rewrite.register
    def _(self, expr: ir.Max):
        values = tuple(self.lookup(value) for value in expr.values)
        return ir.Max(values)

    @_rewrite.register
    def _(self, expr: ir.Slice):
        start = self.lookup(expr.start)
        stop = self.lookup(expr.stop)
        step = self.lookup(expr.step)
        return ir.Slice(start, stop, step)

    @_rewrite.register
    def _(self, expr: ir.Tuple):
        elements = tuple(self.lookup(elem) for elem in expr.elements)
        return ir.Tuple(elements)

    @_rewrite.register
    def _(self, expr: ir.Ternary):
        test = self.lookup(expr.test)
        if_expr = self.lookup(expr.if_expr)
        else_expr = self.lookup(expr.else_expr)
        return ir.Ternary(test, if_expr, else_expr)

    @_rewrite.register
    def _(self, expr: ir.BinOp):
        left = self.lookup(expr.left)
        right = self.lookup(expr.right)
        return ir.BinOp(left, right, expr.op)

    @_rewrite.register
    def _(self, expr: ir.Call):
        args = tuple(self.lookup(arg) for arg in expr.args)
        kws = tuple((kw, self.lookup(value)) for (kw, value) in expr.keywords)
        func = self.lookup(expr.func)
        return ir.Call(func, args, kws)

    @_rewrite.register
    def _(self, expr: ir.BoolOp):
        operands = tuple(self.lookup(operand) for operand in expr.operands)
        return ir.BoolOp(operands, expr.op)

    @_rewrite.register
    def _(self, expr: ir.AffineSeq):
        start = self.lookup(expr.start)
        stop = self.lookup(expr.stop)
        step = self.lookup(expr.step)
        return ir.AffineSeq(start, stop, step)

    @_rewrite.register
    def _(self, expr: ir.UnaryOp):
        operand = self.lookup(expr.operand)
        return ir.UnaryOp(operand, expr.op)

    @_rewrite.register
    def _(self, expr: ir.Zip):
        elems = tuple(self.lookup(elem) for elem in expr.elements)
        return ir.Zip(elems)

    @_rewrite.register
    def _(self, expr: ir.Reversed):
        iterable = self.lookup(expr.iterable)
        return ir.Reversed(iterable)

    @_rewrite.register
    def _(self, expr: ir.IntConst):
        return expr

    @_rewrite.register
    def _(self, expr: ir.FloatConst):
        return expr

    @_rewrite.register
    def _(self, expr: ir.BoolConst):
        return expr

    @_rewrite.register
    def _(self, expr: ir.NameRef):
        return expr


class StmtVisitor:

    @singledispatchmethod
    def visit(self, node):
        msg = f"No handler exists for node type {type(node)}."
        raise NotImplementedError(msg)

    @visit.register
    def _(self, node: list):
        for stmt in node:
            self.visit(stmt)
            # ignore unreachable
            if isinstance(stmt, (ir.Continue, ir.Break, ir.Return)):
                break

    @visit.register
    def _(self, node: ir.Function):
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.IfElse):
        self.visit(node.if_branch)
        self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.ForLoop):
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.WhileLoop):
        self.visit(node.test)
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.Assign):
        pass

    @visit.register
    def _(self, node: ir.SingleExpr):
        pass

    @visit.register
    def _(self, node: ir.Continue):
        pass

    @visit.register
    def _(self, node: ir.Break):
        pass


class StmtTransformer:

    @singledispatchmethod
    def visit(self, node):
        msg = f"No handler for node type {type(node)}."
        raise NotImplementedError(msg)

    @visit.register
    def _(self, node: list):
        repl = []
        for stmt in node:
            stmt = self.visit(stmt)
            repl.append(stmt)
        if repl != node:
            # Only return a copy if it differs from the input.
            node = repl
        return node

    @visit.register
    def _(self, node: ir.Function):
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.IfElse):
        if_branch = self.visit(node.if_branch)
        else_branch = self.visit(node.else_branch)
        if if_branch is not node.if_branch or else_branch is not node.else_branch:
            node = ir.IfElse(node.test, if_branch, else_branch, node.pos)
        return node

    @visit.register
    def _(self, node: ir.ForLoop):
        body = self.visit(node.body)
        if body != node.body:
            node = ir.ForLoop(node.target, node.iterable, body, node.pos)
        return node

    @visit.register
    def _(self, node: ir.WhileLoop):
        body = self.visit(node.body)
        if body is not node.body:
            node = ir.WhileLoop(node.test, body, node.pos)
        return node

    @visit.register
    def _(self, node: ir.Assign):
        return node

    @visit.register
    def _(self, node: ir.SingleExpr):
        return node

    @visit.register
    def _(self, node: ir.Continue):
        return node

    @visit.register
    def _(self, node: ir.Break):
        return node
