import itertools
import typing
from dataclasses import dataclass
from functools import singledispatch, singledispatchmethod

import ir


# statements walked based on depth first ordering
# walking expressions uses post ordering of sub-expressions

@singledispatch
def walk(node):
    raise NotImplementedError


@walk.register
def _(node: ir.Module):
    for node in itertools.chain(node.imports, node.funcs):
        yield node


@walk.register
def _(node: ir.Function):
    for stmt in node.body:
        yield stmt


@walk.register
def _(node: ir.IfElse):
    for stmt in itertools.chain(node.if_branch, node.else_branch):
        yield stmt


@walk.register
def _(node: ir.ForLoop):
    for stmt in node.body:
        yield stmt


@walk.register
def _(node: ir.WhileLoop):
    for stmt in node.body:
        yield stmt


@walk.register
def _(node: ir.Expression):
    """
    Generate post order walk of sub-expressions, ignoring duplicates.

    """

    seen = set()
    queued: typing.List[typing.Tuple[typing.Optional[ir.Expression], typing.Generator]] = [(None, node.subexprs)]

    while queued:
        try:
            e = next(queued[-1][1])
            if e in seen:
                continue
            seen.add(e)
            if e.is_expr:
                queued.append((e, e.subexprs))
            else:
                yield e
        except StopIteration:
            e, _ = queued.pop()
            # ignore the original node we're walking
            if queued:
                yield e


def walk_unpacking(target, value, is_iterated=True):
    def is_unpackable(t_, v_):
        if is_iterated:
            return (isinstance(t_, ir.Tuple)
                    and isinstance(v_, ir.Zip)
                    and len(t_.elements) == len(v_.elements))
        else:
            return (isinstance(t_, ir.Tuple)
                    and isinstance(v_, ir.Tuple)
                    and len(t_.elements) == len(v_.elements))

    if is_unpackable(target, value):
        queued = [zip(target.elements, value.elements)]
        while queued:
            try:
                t, v = next(queued[-1])
                if is_unpackable(t, v):
                    queued.append(zip(t.elements, v.elements))
                else:
                    yield t, v
            except StopIteration:
                queued.pop()
    else:
        yield target, value


@singledispatch
def walk_assignments(node):
    msg = f"No method to walk assignments for node of type {type(node)}"
    raise NotImplementedError(msg)


@walk_assignments.register
def _(node: ir.Assign):
    if node.is_permutation:
        yield from walk_unpacking(node.target, node.value, is_iterated=False)
    elif isinstance(node.target, ir.Tuple):
        # (a,b,c,d) = some_tuple
        for i, t in enumerate(node.target.elements):
            yield t, ir.Subscript(node.value, ir.IntNode(i))
    else:
        yield node.target, node.value


@walk_assignments.register
def _(node: ir.CascadeAssign):
    for target in node.targets:
        yield target, node.value


@walk_assignments.register
def _(node: ir.ForLoop):
    return walk_unpacking(node.target, node.iterable, is_iterated=True)


def walk_multiple(exprs):
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
        if expr.is_expr:
            queued.append((expr, expr.subexprs))
        else:
            yield expr
        while queued:
            try:
                e = next(queued[-1][1])
                if e in seen:
                    continue
                seen.add(e)
                if e.is_expr:
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
    def _(self, node: ir.Module):
        imports = node.imports
        funcs = [self.visit(f) for f in node.funcs]
        return ir.Module(funcs, imports)

    @visit.register
    def _(self, node: ir.Function):
        name = node.name
        self.visit(node.body)
        args = node.args
        body = self.visit(node.body)
        types = node.types
        # compute exact arrays in a later pass
        return ir.Function(name, args, body, types, [])

    @visit.register
    def _(self, node: list):
        for stmt in node:
            self.visit(stmt)

    @visit.register
    def _(self, node: ir.StmtBase):
        pass

    @visit.register
    def _(self, node: ir.Assign):
        self.visit(node.target)
        self.visit(node.value)

    @visit.register
    def _(self, node: ir.CascadeAssign):
        self.visit(node.value)
        for t in node.targets:
            self.visit(t)

    @visit.register
    def _(self, node: ir.SingleExpr):
        self.visit(node.expr)

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
    def _(self, node: ir.IfElse):
        self.visit(node.test)
        self.visit(node.if_branch)
        self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.Return):
        if node.value is not None:
            self.visit(node.value)

    @visit.register
    def _(self, node: ir.Expression):
        pass

    @visit.register
    def _(self, node: ir.ObjectBase):
        pass

    @visit.register
    def _(self, node: ir.Subscript):
        self.visit(node.value)
        self.visit(node.slice)

    @visit.register
    def _(self, node: ir.SimpleSlice):
        self.visit(node.start)
        self.visit(node.stop)
        self.visit(node.step)

    @visit.register
    def _(self, node: ir.BinOp):
        self.visit(node.left)
        self.visit(node.right)

    @visit.register
    def _(self, node: ir.BoolOp):
        for operand in node.operands:
            self.visit(operand)

    @visit.register
    def _(self, node: ir.Argument):
        pass

    @visit.register
    def _(self, node: ir.NameRef):
        pass

    @visit.register
    def _(self, node: ir.Call):
        for arg in node.args:
            self.visit(arg)

    @visit.register
    def _(self, node: ir.CompareOp):
        for operand in node.operands:
            self.visit(operand)

    @visit.register
    def _(self, node: ir.Counter):
        self.visit(node.start)
        self.visit(node.stop)
        self.visit(node.step)

    @visit.register
    def _(self, node: ir.IfExpr):
        self.visit(node.test)
        self.visit(node.if_expr)
        self.visit(node.else_expr)

    @visit.register
    def _(self, node: ir.Reversed):
        self.visit(node.iterable)

    @visit.register
    def _(self, node: ir.Tuple):
        for e in node.elements:
            self.visit(e)

    @visit.register
    def _(self, node: ir.UnaryOp):
        self.visit(node.operand)

    @visit.register
    def _(self, node: ir.Cast):
        self.visit(node.expr)

    @visit.register
    def _(self, node: ir.Zip):
        for e in node.elements:
            self.visit(e)


@dataclass(frozen=True)
class AssignPair:
    """
    Helper to maintain context of serialized assignment pairs.

    """

    target: typing.Union[ir.ObjectBase, ir.Subscript]
    value: typing.Union[ir.ObjectBase, ir.Expression]
    iterated: bool


class AssignCollector(VisitorBase):
    """
    Resolves expression types, given free variable types

    """

    def __call__(self, entry):
        self.assigned_order = []
        self.truth_tested = set()
        self.return_values = set()
        self.visit(entry)
        assigned_order = self.assigned_order
        truth_tested = self.truth_tested
        return_values = self.return_values
        self.assigned_order = self.truth_tested = self.return_values = None
        return assigned_order, truth_tested, return_values

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.ForLoop):
        for target, value in walk_assignments(node):
            # this is accepted by CPython but should raise an error
            # elsewhere in this code due to potential cascading conflicts
            self.assigned_order.append(AssignPair(target, value, iterated=True))

    @visit.register
    def _(self, node: ir.IfElse):
        if isinstance(node.test, ir.BoolNode):
            self.visit(node.test)
        else:
            self.truth_tested.add(node.test)
        self.visit(node.if_branch)
        self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.WhileLoop):
        if isinstance(node.test, ir.BoolNode):
            self.visit(node.test)
        else:
            self.truth_tested.add(node.test)
        self.truth_tested.add(node.test)
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.Assign):
        for target, value in walk_assignments(node):
            if target.is_expr:
                continue
            self.assigned_order.append(AssignPair(target, value, iterated=False))

    @visit.register
    def _(self, node: ir.CascadeAssign):
        for target, value in walk_assignments(node):
            if target.is_expr:
                # this is accepted by CPython but should raise an error
                # elsewhere in this code due to potential cascading conflicts
                continue
            self.assigned_order.append(AssignPair(target, value, iterated=False))

    @visit.register
    def _(self, node: ir.Expression):
        # can't do anything here
        pass

    @visit.register
    def _(self, node: ir.BoolOp):
        for operand in node.operands:
            if isinstance(operand, ir.BoolOp):
                self.visit(operand)
            else:
                self.truth_tested.add(operand)

    @visit.register
    def _(self, node: ir.Return):
        self.visit(node.value)
        self.return_values.add(node.value)


class TransformBase:
    """
    This is intended to rebuild everything by default.
    """

    def __call__(self, node):
        return self.visit(node)

    @singledispatchmethod
    def visit(self, node):
        msg = f"No handler for node type {type(node)}"
        raise NotImplementedError(msg)

    @visit.register
    def _(self, node: ir.Module):
        imports = node.imports
        funcs = [self.visit(f) for f in node.funcs]
        return ir.Module(funcs, imports)

    @visit.register
    def _(self, node: ir.Function):
        name = node.name
        args = node.args
        body = self.visit(node.body)
        types = node.types
        # compute exact arrays in a later pass
        return ir.Function(name, args, body, types, [])

    @visit.register
    def _(self, node: list):
        return [self.visit(stmt) for stmt in node]

    @visit.register
    def _(self, node: ir.StmtBase):
        return node

    @visit.register
    def _(self, node: ir.Assign):
        node.target = self.visit(node.target)
        node.value = self.visit(node.value)
        return node

    @visit.register
    def _(self, node: ir.CascadeAssign):
        node.value = self.visit(node.value)
        node.targets = [self.visit(t) for t in node.targets]
        return node

    @visit.register
    def _(self, node: ir.SingleExpr):
        node.expr = self.visit(node.expr)
        return node

    @visit.register
    def _(self, node: ir.ForLoop):
        iterable = node.iterable
        target = node.target
        body = self.visit(node.body)
        pos = node.pos
        return ir.ForLoop(iterable, target, body, pos)

    @visit.register
    def _(self, node: ir.WhileLoop):
        test = node.test
        body = self.visit(node.body)
        pos = node.pos
        return ir.WhileLoop(test, body, pos)

    @visit.register
    def _(self, node: ir.IfElse):
        test = node.test
        if_branch = self.visit(node.if_branch)
        else_branch = self.visit(node.else_branch)
        pos = node.pos
        return ir.IfElse(test, if_branch, else_branch, pos)

    @visit.register
    def _(self, node: ir.Return):
        node.value = self.visit(node.value) if node.value else None
        return node

    @visit.register
    def _(self, node: ir.Expression):
        return node

    @visit.register
    def _(self, node: ir.ObjectBase):
        return node

    @visit.register
    def _(self, node: ir.Subscript):
        return ir.Subscript(self.visit(node.value), self.visit(node.slice))

    @visit.register
    def _(self, node: ir.SimpleSlice):
        return ir.SimpleSlice(self.visit(node.start), self.visit(node.stop), self.visit(node.step))

    @visit.register
    def _(self, node: ir.BinOp):
        return ir.BinOp(self.visit(node.left), self.visit(node.right), node.op)

    @visit.register
    def _(self, node: ir.BoolOp):
        return ir.BoolOp(tuple(self.visit(operand) for operand in node.operands), node.op)

    @visit.register
    def _(self, node: ir.Argument):
        return node

    @visit.register
    def _(self, node: ir.NameRef):
        return node

    @visit.register
    def _(self, node: ir.Call):
        args = tuple(self.visit(arg) for arg in node.args)
        keywords = tuple((kw, self.visit(value)) for (kw, value) in node.keywords)
        return ir.Call(node.funcname, args, keywords)

    @visit.register
    def _(self, node: ir.CompareOp):
        return ir.CompareOp(tuple(self.visit(operand) for operand in node.operands), node.ops)

    @visit.register
    def _(self, node: ir.Counter):
        start = self.visit(node.start)
        stop = self.visit(node.stop) if node.stop is not None else None
        step = self.visit(node.step)
        return ir.Counter(start, stop, step)

    @visit.register
    def _(self, node: ir.IfExpr):
        return ir.IfExpr(self.visit(node.test), self.visit(node.if_expr), self.visit(node.else_expr))

    @visit.register
    def _(self, node: ir.Reversed):
        return ir.Reversed(self.visit(node.iterable))

    @visit.register
    def _(self, node: ir.Tuple):
        return ir.Tuple(tuple(self.visit(e) for e in node.elements))

    @visit.register
    def _(self, node: ir.UnaryOp):
        return ir.UnaryOp(self.visit(node.operand), node.op)

    @visit.register
    def _(self, node: ir.Cast):
        return ir.Cast(self.visit(node.expr), node.as_type)

    @visit.register
    def _(self, node: ir.Zip):
        return ir.Zip(tuple(self.visit(e) for e in node.elements))
