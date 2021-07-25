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


class RuleBasedRewriter:
    """
    A small utility to help cache expression mappings and rewrite expressions according to
    updates of their sub-expressions.

    """

    def __init__(self):
        self.rules = {}
        self.cache = {}

    def clear(self, expr=None):
        if expr is None:
            self.rules.clear()
        else:
            self.rules.pop(expr, None)

    def update_rule(self, expr, output):
        assert output is not None
        curr = self.rules.get(expr)
        if curr is None or curr != output:
            self.rules[expr] = output

    def make_rule(self, expr, output):
        assert output is not None
        curr = self.rules.get(expr)
        if curr is not None:
            if curr != output:
                msg = f"Already have a non-matching existing rule for {expr}."
                raise KeyError(msg)
        else:
            self.rules[expr] = output

    def lookup(self, expr):
        output = self.rules.get(expr)
        if output is None:
            output = expr
        return output

    def reconstruct_expression(self, expr):
        """
        Rewrites an expression based on rules available to its sub-expressions.
        """

        # if we have a rule set for this expression,
        # it takes priority
        from_cache = self.rules.get(expr)
        if from_cache is None:
            # Check if previously reconstructed
            form_cache = self.cache.get(expr)
            if from_cache is None:
                from_cache = self._rewrite(expr)
                if from_cache == expr:
                    # if they match, use the original expression
                    from_cache = expr
                self.cache[expr] = form_cache
        return from_cache

    @singledispatchmethod
    def _rewrite(self, expr):
        raise NotImplementedError

    @_rewrite.register
    def _(self, expr: ir.Length):
        value = self.lookup(expr.value)
        repl = ir.Length(value)
        return repl

    @_rewrite.register
    def _(self, expr: ir.Subscript):
        value = self._rewrite(expr.value)
        slice_ = self._rewrite(expr.slice)
        subscript = ir.Subscript(value, slice_)
        return subscript

    @_rewrite.register
    def _(self, expr: ir.Min):
        output = tuple(self.lookup(elem) for elem in expr.values)
        return output

    @_rewrite.register
    def _(self, expr: ir.Max):
        output = tuple(self.rules[elem] for elem in expr.values)
        return output

    @_rewrite.register
    def _(self, expr: ir.Slice):
        start = self.lookup(expr.start)
        stop = self.lookup(expr.stop)
        step = self.lookup(expr.step)
        return ir.Slice(start, stop, step)

    @_rewrite.register
    def _(self, expr: ir.Tuple):
        output = tuple(self.lookup(elem) for elem in expr.elements)
        output = ir.Tuple(output)
        return output

    @_rewrite.register
    def _(self, expr: ir.Ternary):
        test = self.lookup(expr.test)
        on_true = self.lookup(expr.if_expr)
        on_false = self.lookup(expr.else_expr)
        output = ir.Ternary(test, on_true, on_false)
        return output

    @_rewrite.register
    def _(self, expr: ir.BinOp):
        left = self.lookup(expr.left)
        right = self.lookup(expr.right)
        output = ir.BinOp(left, right, expr.op)
        return output

    @_rewrite.register
    def _(self, expr: ir.Call):
        args = tuple(self.lookup(arg) for arg in expr.args)
        kws = tuple((kw, self.lookup(value)) for (kw, value) in expr.keywords)
        func = self.lookup(expr.func)
        output = ir.Call(func, args, kws)
        return output

    @_rewrite.register
    def _(self, expr: ir.BoolOp):
        subexprs = tuple(self.lookup(operand) for operand in expr.operands)
        output = ir.BoolOp(subexprs, expr.op)
        return output

    @_rewrite.register
    def _(self, expr: ir.AffineSeq):
        start = self.lookup(expr.start)
        stop = self.lookup(expr.stop)
        step = self.lookup(expr.step)
        output = ir.AffineSeq(start, stop, step)
        return output

    @_rewrite.register
    def _(self, expr: ir.UnaryOp):
        operand = self.lookup(expr.operand)
        output = ir.UnaryOp(operand, expr.op)
        return output

    @_rewrite.register
    def _(self, expr: ir.Zip):
        elems = tuple(self.lookup(elem) for elem in expr.elements)
        output = ir.Zip(elems)
        return output

    @_rewrite.register
    def _(self, expr: ir.Reversed):
        iterable = self.lookup(expr.iterable)
        output = ir.Reversed(iterable)
        return output

    @_rewrite.register
    def _(self, expr: ir.IntNode):
        return expr

    @_rewrite.register
    def _(self, expr: ir.FloatNode):
        return expr

    @_rewrite.register
    def _(self, expr: ir.BoolNode):
        return expr

    @_rewrite.register
    def _(self, expr: ir.NameRef):
        return expr


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
