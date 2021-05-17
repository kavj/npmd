from functools import singledispatchmethod

import ir


# Walkers are named by what they can walk.

def walk(node):
    """
    extending walk interface to include lists
    """
    if isinstance(node, list):
        for stmt in node:
            yield stmt
    else:
        yield from node.walk()


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


def walk_all(node):
    """
    Walk everything, expanding branches and loop constructs
    Ignore nested scopes

    """

    # don't yield entry node
    if isinstance(node, ir.ForLoop):
        for assign in node.assigns:
            yield assign
    if isinstance(node, ir.Walkable):
        for stmt in node.walk():
            yield stmt
            if isinstance(stmt, ir.Walkable):
                yield from walk_all(stmt)


def walk_expressions(exprs):
    """
    This walks an iterable of expression nodes, each in post order, ignoring duplicates.
    It's assumed that iter(exprs) yields a safe ordering.

    """

    queued = []
    seen = set()
    for expr in exprs:
        if expr in seen:
            continue
        seen.add(expr)
        if isinstance(expr, ir.Expression):
            queued.append((expr, expr.subexprs))
        else:
            yield expr
        while queued:
            try:
                e = next(queued[-1][1])
                if e in seen:
                    continue
                seen.add(e)
                if isinstance(e, ir.Expression):
                    queued.append(e)
                else:
                    yield e
            except StopIteration:
                e, _ = queued.pop()
                yield e


class VisitorBase:

    @singledispatchmethod
    def visit(self, node):
        msg = f"No handler for node type {type(node)}"
        raise NotImplementedError(msg)

    @visit.register
    def _(self, node: list):
        for stmt in node:
            self.visit(stmt)

    @visit.register
    def _(self, node: ir.Module):
        for func in node.funcs:
            self.visit(func)

    @visit.register
    def _(self, node: ir.Function):
        for stmt in node.walk():
            self.visit(stmt)

    @visit.register
    def _(self, node: ir.IfElse):
        self.visit(node.test)
        self.visit(node.if_branch)
        self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.ForLoop):
        for _, iterable in node.assigns:
            self.visit(iterable)
        for target, _ in node.assigns:
            self.visit(target)
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.WhileLoop):
        self.visit(node.test)
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.Assign):
        self.visit(node.target)
        self.visit(node.value)

    @visit.register
    def _(self, node: ir.SingleExpr):
        self.visit(node.expr)

    @visit.register
    def _(self, node: ir.StmtBase):
        pass

    @visit.register
    def _(self, node: ir.Expression):
        for subexpr in node.subexprs:
            self.visit(subexpr)


class TransformBase:
    """
    Since expressions are immutable, this class provides default methods of reconstruction,
    in case a sub-expression changes.

    """

    @singledispatchmethod
    def visit(self, node):
        msg = f"No handler for node type {type(node)}"
        raise NotImplementedError(msg)

    @visit.register
    def _(self, node: list):
        repl = []
        for stmt in node:
            repl_stmt = self.visit(stmt)
            repl.append(repl_stmt)
        return repl

    @visit.register
    def _(self, node: ir.Module):
        imports = node.imports
        funcs = [self.visit(f) for f in node.funcs]
        return ir.Module(funcs, imports)

    @visit.register
    def _(self, node: ir.Function):
        name = node.name
        repl = self.visit(node.body)
        types = node.types
        arrays = node.arrays
        # compute exact arrays in a later pass
        return ir.Function(name, node.args, repl, types, arrays)

    @visit.register
    def _(self, node: ir.Assign):
        target = self.visit(node.target)
        value = self.visit(node.value)
        pos = node.pos
        repl = ir.Assign(target, value, pos)
        return repl

    @visit.register
    def _(self, node: ir.ForLoop):
        assigns = []
        for target, value in node.assigns:
            assigns.append((self.visit(target), self.visit(value)))
        repl_body = self.visit(node.body)
        pos = node.pos
        return ir.ForLoop(assigns, repl_body, pos)

    @visit.register
    def _(self, node: ir.WhileLoop):
        test = self.visit(node.test)
        repl = self.visit(node.body)
        pos = node.pos
        return ir.WhileLoop(test, repl, pos)

    @visit.register
    def _(self, node: ir.IfElse):
        test = self.visit(node.test)
        if_branch = self.visit(node.if_branch)
        else_branch = self.visit(node.else_branch)
        pos = node.pos
        return ir.IfElse(test, if_branch, else_branch, pos)

    @visit.register
    def _(self, node: ir.Continue):
        return node

    @visit.register
    def _(self, node: ir.Pass):
        return node

    @visit.register
    def _(self, node: ir.Break):
        return node

    @visit.register
    def _(self, node: ir.SingleExpr):
        expr = self.visit(node.expr)
        repl = ir.SingleExpr(expr, node.pos)
        return repl

    @visit.register
    def _(self, node: ir.Return):
        value = self.visit(node.value) if node.value is not None else None
        return ir.Return(value, node.pos)

    @visit.register
    def _(self, node: ir.Expression):
        return node

    @visit.register
    def _(self, node: ir.Subscript):
        return ir.Subscript(self.visit(node.value), self.visit(node.slice))

    @visit.register
    def _(self, node: ir.Slice):
        return ir.Slice(self.visit(node.start), self.visit(node.stop), self.visit(node.step))

    @visit.register
    def _(self, node: ir.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op
        repl = ir.BinOp(left, right, op)
        return repl

    @visit.register
    def _(self, node: ir.BoolOp):
        operands = tuple(self.visit(operand) for operand in node.operands)
        op = node.op
        repl = ir.BoolOp(operands, op)
        return repl

    @visit.register
    def _(self, node: ir.Argument):
        return node

    @visit.register
    def _(self, node: ir.NameRef):
        return node

    @visit.register
    def _(self, node: ir.Constant):
        return node

    @visit.register
    def _(self, node: ir.Call):
        args = tuple(self.visit(arg) for arg in node.args)
        keywords = tuple((kw, self.visit(value)) for (kw, value) in node.keywords)
        return ir.Call(node.funcname, args, keywords)

    @visit.register
    def _(self, node: ir.Counter):
        start = self.visit(node.start)
        stop = self.visit(node.stop) if node.stop is not None else None
        step = self.visit(node.step)
        return ir.Counter(start, stop, step)

    @visit.register
    def _(self, node: ir.IfExpr):
        test = self.visit(node.test)
        if_expr = self.visit(node.if_expr)
        else_expr = self.visit(node.else_expr)
        repl = ir.IfExpr(test, if_expr, else_expr)
        return repl

    @visit.register
    def _(self, node: ir.Reversed):
        iterable = self.visit(node.iterable)
        repl = ir.Reversed(iterable)
        return repl

    @visit.register
    def _(self, node: ir.Tuple):
        repl = ir.Tuple(tuple(self.visit(e) for e in node.elements))
        return repl

    @visit.register
    def _(self, node: ir.UnaryOp):
        operand = self.visit(node.operand)
        repl = ir.UnaryOp(operand, node.op)
        return repl

    @visit.register
    def _(self, node: ir.Cast):
        return ir.Cast(self.visit(node.expr), node.as_type)
