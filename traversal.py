import inspect
import itertools
from functools import singledispatchmethod
from typing import Callable, Generator, Iterator, List, Tuple

import ir

from errors import CompilerError
from utils import is_entry_point


def sequence_block_intervals(stmts: List[ir.StmtBase]):
    segment_first = 0
    segment_last = 0
    for segment_last, stmt in enumerate(stmts):
        if is_entry_point(stmt):
            if segment_first < segment_last:
                # yield up to this statement
                yield segment_first, segment_last
            yield segment_last, segment_last + 1
            segment_first = segment_last + 1
    if segment_first <= segment_last and segment_first < len(stmts):
        yield segment_first, segment_last + 1


def sequence_blocks(stmts: List[ir.StmtBase]):
    segment_first = 0
    segment_last = 0
    for segment_last, stmt in enumerate(stmts):
        if is_entry_point(stmt):
            if segment_first < segment_last:
                # yield up to this statement
                yield itertools.islice(stmts, segment_first, segment_last)
            yield itertools.islice(stmts, segment_last, segment_last + 1)
            segment_first = segment_last + 1
    if segment_first <= segment_last:
        yield itertools.islice(stmts, segment_first, segment_last + 1)


def depth_first_sequence_statements(node: List[ir.StmtBase], reverse=False) -> Generator[ir.StmtBase, None, None]:
    """

    :param node:
    :param reverse:
    :return:
    """
    iterator = reversed if reverse else iter

    node_iter = iterator(node)
    queued = [node_iter]
    while queued:
        node_iter = queued[-1]
        for stmt in node_iter:
            yield stmt
            if is_entry_point(stmt):
                if isinstance(stmt, ir.IfElse):
                    if stmt.else_branch:
                        queued.append(iterator(stmt.else_branch))
                    if stmt.if_branch:
                        queued.append(iterator(stmt.if_branch))
                else:
                    assert isinstance(stmt, (ir.ForLoop, ir.WhileLoop))
                    queued.append(iterator(stmt.body))
                break
        else:
            # iterator exhausted
            queued.pop()


def depth_first_sequence_blocks(node: List[ir.StmtBase]) -> Generator[Iterator[ir.StmtBase], None, None]:
    """
    basic block generator
    :param node:
    :return:
    """

    block_iter = sequence_block_intervals(node)
    queued = [(node, block_iter)]
    while queued:
        stmts, block_iter = queued[-1]
        for start, stop in block_iter:
            yield itertools.islice(stmts, start, stop)
            if stop - start == 1:
                first = stmts[start]
                if is_entry_point(first):
                    if isinstance(first, ir.IfElse):
                        if first.else_branch:
                            queued.append((first.else_branch, sequence_block_intervals(first.else_branch)))
                        if first.if_branch:
                            queued.append((first.if_branch, sequence_block_intervals(first.if_branch)))
                    else:
                        assert isinstance(first, (ir.ForLoop, ir.WhileLoop))
                        queued.append((first.body, sequence_block_intervals(first.body)))
                    break
        else:
            # iterator exhausted
            queued.pop()


def walk(node: ir.ValueRef):
    """
    yields all distinct sub-expressions and the base expression
    It was changed to include the base expression so that it's safe
    to walk without the need for explicit dispatch mechanics in higher level
    functions on Expression to disambiguate Expression vs non-Expression value
    references.

    :param node:
    :return:
    """
    if not isinstance(node, ir.ValueRef):
        msg = f'walk expects a value ref. Received: "{node}"'
        raise TypeError(msg)
    assert isinstance(node, ir.ValueRef)
    if isinstance(node, ir.Expression):
        seen = {node}
        enqueued = [(node, node.subexprs)]
        while enqueued:
            expr, subexprs = enqueued[-1]
            for subexpr in subexprs:
                if subexpr in seen:
                    continue
                seen.add(subexpr)
                if isinstance(subexpr, ir.Expression):
                    enqueued.append((subexpr, subexpr.subexprs))
                    break
                yield subexpr
            else:
                # exhausted
                yield expr
                enqueued.pop()
    else:
        yield node


def array_safe_walk(node: ir.ValueRef):
    """
    like walk, but doesn't follow subscript.value paths.

    For example, walking 'a[i]' should yield ir versions of 'a', 'i', 'a[i]'. Here we would not yield 'a'.

    For a nested subscript like a[i][j] (once properly supported) this would be updated to check follow the nested
    index paths. For now, nested subscripts are unsupported (as the interplay with vectorization is not trivial).

    :param node:
    :return:
    """

    assert isinstance(node, ir.ValueRef)
    if isinstance(node, ir.Expression):
        seen = {node}
        if isinstance(node, ir.Subscript):
            # don't yield node.value, since this would suggest
            # a bare array reference rather than a subscripted one
            enqueued = [(node, iter((node.index,)))]
        else:
            enqueued = [(node, node.subexprs)]
        while enqueued:
            expr, subexprs = enqueued[-1]
            for subexpr in subexprs:
                if subexpr in seen:
                    continue
                seen.add(subexpr)
                if isinstance(subexpr, ir.Expression):
                    if isinstance(subexpr, ir.Subscript):
                        # don't descend on node.value, so on encountering expression a[i],
                        # yield i then a[i] rather than a, i, a[i], since we only access an element of 'a'.
                        enqueued.append((subexpr, iter((node.index,))))
                    else:
                        enqueued.append((subexpr, subexpr.subexprs))
                    break
                yield subexpr
            else:
                # exhausted
                yield expr
                enqueued.pop()
    else:
        yield node


def walk_parameters(node: ir.ValueRef):
    for value in walk(node):
        if isinstance(value, ir.NameRef):
            yield value


def get_value_types(node: ir.ValueRef, types: Tuple):
    """
    filtering version of walk
    :param node:
    :param types:
    :return:
    """
    for value in walk(node):
        if isinstance(value, types):
            yield value


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
    def _(self, node: ir.StmtBase):
        pass

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


class BlockRewriter:
    def __init__(self, transform: Callable):
        self.transform = transform

    @singledispatchmethod
    def visit(self, node):
        msg = f"No handler for node type {type(node)}."
        raise NotImplementedError(msg)

    @visit.register
    def _(self, node: list):
        repl = []
        for start, stop in sequence_block_intervals(node):
            if is_entry_point(node[start]):
                assert stop - start == 1
                stmt = self.visit(node[start])
                repl.append(stmt)
            else:
                block = self.transform(node[start: stop])
                repl.extend(block)
        return repl

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
    def _(self, node: ir.StmtBase):
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
