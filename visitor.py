from functools import singledispatchmethod

import ir


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


def walk_parameters(node):
    """
    Walk an expression, yielding only sub-expressions that are not expressions themselves.

    """
    if not isinstance(node, ir.ValueRef):
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
                if isinstance(expr, ir.ValueRef):
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
            queued.append(expr)
            queued.extend(expr.subexprs)


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


class StmtVisitor:

    @singledispatchmethod
    def visit(self, node):
        msg = f"No handler for node type {type(node)}."
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
            # ignore unreachable
            if isinstance(stmt, (ir.Continue, ir.Break, ir.Return)):
                break
        return repl

    @visit.register
    def _(self, node: ir.Function):
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.IfElse):
        self.visit(node.if_branch)
        self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.ForLoop):
        body = self.visit(node.body)
        repl = ir.ForLoop(node.target, node.iterable, body, node.pos)
        return repl

    @visit.register
    def _(self, node: ir.WhileLoop):
        body = self.visit(node.body)
        repl = ir.WhileLoop(node.test, body, node.pos)
        return repl

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
