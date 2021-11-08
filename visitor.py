from functools import singledispatchmethod, lru_cache

import ir


def walk(node):
    """
    walk an expression depth first in post order, yielding everything but the original node
    """
    if hasattr(node, "subexprs"):
        for subexpr in node.subexprs:
            yield from walk(subexpr)
            yield subexpr


def walk_unique(node):
    seen = set()
    for subexpr in walk(node):
        if subexpr not in seen:
            yield subexpr
            seen.add(subexpr)


class ExpressionVisitor:
    """
    Since expressions are immutable and hashable, we can cache their results.

    """

    def invalidate_cache(self):
        self.lookup.cache_clear()

    @lru_cache
    def lookup(self, expr):
        repl = self.visit(expr)
        # prefer the original rather over an identical copy
        return expr if expr == repl else repl

    # Don't cache visitor results directly, since
    # they may be overridden.

    @singledispatchmethod
    def visit(self, expr):
        msg = f"No method to rewrite object of type {type(expr)}."
        raise NotImplementedError(msg)

    @visit.register
    def _(self, expr: ir.SingleDimRef):
        return self.visit(base)

    @visit.register
    def _(self, expr: ir.SingleDimRef):
        return self.visit(expr.base)

    @visit.register
    def _(self, expr: ir.Subscript):
        value = self.lookup(expr.value)
        slice_ = self.lookup(expr.slice)
        return ir.Subscript(value, slice_)

    @visit.register
    def _(self, expr: ir.Min):
        values = tuple(self.lookup(value) for value in expr.values)
        return ir.Min(values)

    @visit.register
    def _(self, expr: ir.Max):
        values = tuple(self.lookup(value) for value in expr.values)
        return ir.Max(values)

    @visit.register
    def _(self, expr: ir.Slice):
        start = self.lookup(expr.start)
        stop = self.lookup(expr.stop)
        step = self.lookup(expr.step)
        return ir.Slice(start, stop, step)

    @visit.register
    def _(self, expr: ir.Tuple):
        elements = tuple(self.lookup(elem) for elem in expr.elements)
        return ir.Tuple(elements)

    @visit.register
    def _(self, expr: ir.Select):
        predicate = self.lookup(expr.predicate)
        on_true = self.lookup(expr.on_true)
        on_false = self.lookup(expr.on_false)
        return ir.Select(on_true, on_false, predicate)

    @visit.register
    def _(self, expr: ir.BinOp):
        left = self.lookup(expr.left)
        right = self.lookup(expr.right)
        return ir.BinOp(left, right, expr.op)

    @visit.register
    def _(self, expr: ir.CompareOp):
        left = self.lookup(expr.left)
        right = self.lookup(expr.right)
        return ir.CompareOp(left, right, expr.op)

    @visit.register
    def _(self, expr: ir.Call):
        args = tuple(self.lookup(arg) for arg in expr.args)
        kws = tuple((kw, self.lookup(value)) for (kw, value) in expr.keywords)
        func = self.lookup(expr.func)
        return ir.Call(func, args, kws)

    @visit.register
    def _(self, expr: ir.AND):
        operands = tuple(self.lookup(operand) for operand in expr.operands)
        return ir.AND(operands)

    @visit.register
    def _(self, expr: ir.OR):
        operands = tuple(self.lookup(operand) for operand in expr.operands)
        return ir.OR(operands)

    @visit.register
    def _(self, expr: ir.XOR):
        operands = tuple(self.lookup(operand) for operand in expr.operands)
        return ir.XOR(operands)

    @visit.register
    def _(self, expr: ir.TRUTH):
        operand = self.lookup(expr.operand)
        return ir.XOR(operand)

    @visit.register
    def _(self, expr: ir.NOT):
        operand = self.lookup(expr.operand)
        return ir.XOR(operand)

    @visit.register
    def _(self, expr: ir.AffineSeq):
        start = self.lookup(expr.start)
        stop = self.lookup(expr.stop)
        step = self.lookup(expr.step)
        return ir.AffineSeq(start, stop, step)

    @visit.register
    def _(self, expr: ir.SingleDimRef):
        base = self.visit(expr.base)
        dim = expr.dim
        return ir.SingleDimRef(base, dim)

    @visit.register
    def _(self, expr: ir.UnaryOp):
        operand = self.lookup(expr.operand)
        return ir.UnaryOp(operand, expr.op)

    @visit.register
    def _(self, expr: ir.Zip):
        elems = tuple(self.lookup(elem) for elem in expr.elements)
        return ir.Zip(elems)

    @visit.register
    def _(self, expr: ir.Enumerate):
        iterable = self.lookup(expr.iterable)
        start = self.lookup(expr.start)
        return ir.Enumerate(iterable, start)

    @visit.register
    def _(self, expr: ir.Reversed):
        iterable = self.lookup(expr.iterable)
        return ir.Reversed(iterable)

    @visit.register
    def _(self, expr: ir.IntConst):
        return expr

    @visit.register
    def _(self, expr: ir.FloatConst):
        return expr

    @visit.register
    def _(self, expr: ir.BoolConst):
        return expr

    @visit.register
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

    @visit.register
    def _(self, node: ir.Return):
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
        body = self.visit(node.body)
        if body != node.body:
            return ir.Function(node.name, node.args, body)
        return node

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

    @visit.register
    def _(self, node: ir.Return):
        return node
