from functools import singledispatchmethod

import ir


# Walkers are named by what they can walk.

def is_control_flow_entry(node):
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


def walk_expr_params(node):
    """
    Walk an expression, yielding only sub-expressions that are not expressions themselves.

    """
    if not isinstance(node, ir.Expression):
        assert not isinstance(node, ir.StmtBase)
        yield node
    else:
        queued = [node]
        seen = set()
        while queued:
            expr = queued.pop()
            if expr in seen:
                continue
            else:
                seen.add(expr)
                if isinstance(expr, ir.Expression):
                    queued.extend(expr.subexprs)
                else:
                    yield expr


def walk_expr(node):
    """
    This walks an expression in post order, yielding everything including the original.
    This avoids having to check whether something is an expression as opposed to a name or constant.
    Declaring a name as an implicit expression produces inconsistent behavior, as we can explicitly
    bind to names.
    """

    if not isinstance(node, ir.Expression):
        assert not isinstance(node, ir.StmtBase)
        yield node
    else:
        queued = [node]
        seen = set()
        sent = set()
        while queued:
            expr = queued.pop()
            if expr in seen:
                if expr not in sent:
                    sent.add(expr)
                    yield expr
            else:
                seen.add(expr)
                if isinstance(expr, ir.Expression):
                    queued.append(expr)
                    queued.extend(expr.subexprs)
                else:
                    yield expr


def walk(node):
    """
    extending walk interface to include lists
    """

    if isinstance(node, list):
        for stmt in node:
            yield stmt
            if is_control_flow_entry(stmt):
                yield from walk(stmt)
    elif isinstance(node, (ir.ForLoop, ir.WhileLoop)):
        yield from walk(node.body)
    elif isinstance(node, ir.IfElse):
        yield from walk(node.if_branch)
        yield from walk(node.else_branch)
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
    def _(self, node: ir.NameRef):
        pass

    @visit.register
    def _(self, node: ir.Constant):
        pass

    @visit.register
    def _(self, node: ir.Module):
        for func in node.funcs:
            self.visit(func)

    @visit.register
    def _(self, node: ir.Function):
        for arg in node.args:
            self.visit(arg)
        for stmt in walk(node.body):
            self.visit(stmt)

    @visit.register
    def _(self, node: ir.IfElse):
        self.visit(node.test)
        self.visit(node.if_branch)
        self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.ForLoop):
        self.visit(node.iterable)
        self.visit(node.target)
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
        # Todo: We may need textwrap and error formatting for some of these.
        #    The change to inline the node itself here is due to lack of information
        #    provided by type info only.
        msg = f"No handler for node {node} of type {type(node)}"
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
        # compute exact arrays in a later pass
        return ir.Function(name, node.args, repl)

    @visit.register
    def _(self, node: ir.Assign):
        target = self.visit(node.target)
        value = self.visit(node.value)
        pos = node.pos
        repl = ir.Assign(target, value, pos)
        return repl

    @visit.register
    def _(self, node: ir.ForLoop):
        iterable = self.visit(node.iterable)
        target = self.visit(node.target)
        repl_body = self.visit(node.body)
        pos = node.pos
        return ir.ForLoop(target, iterable, repl_body, pos)

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
