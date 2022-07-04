from abc import ABC, abstractmethod
from typing import Generator, Iterable, List, Tuple, Union

import ir

from utils import is_entry_point


def sequence_blocks(stmts: Iterable[ir.StmtBase]):
    block = []
    for stmt in stmts:
        if is_entry_point(stmt):
            if block:
                yield block
                block = []
            yield [stmt]
        else:
            block.append(stmt)
    if block:
        yield block


def get_nested(node: Union[ir.IfElse, ir.ForLoop, ir.WhileLoop]):
    if isinstance(node, ir.IfElse):
        yield node.if_branch
        yield node.else_branch
    elif isinstance(node, (ir.ForLoop, ir.WhileLoop)):
        yield node.body


def depth_first_sequence_blocks(node: List[ir.StmtBase]) -> Generator[List[ir.StmtBase], None, None]:
    """
    basic block generator
    :param node:
    :return:
    """

    queued = [sequence_blocks(node)]
    while queued:
        block_iter = queued[-1]
        for block in block_iter:
            yield block
            leading = block[0]
            if isinstance(leading, ir.IfElse):
                if leading.else_branch:
                    queued.append(sequence_blocks(leading.else_branch))
                if leading.if_branch:
                    queued.append(sequence_blocks(leading.if_branch))
                break
            elif isinstance(leading, (ir.ForLoop, ir.WhileLoop)):
                # for or while loop
                if leading.body:
                    queued.append(sequence_blocks(leading.body))
                break  # If any enqueued, we need to visit them next
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


class BlockRewriter(ABC):

    @abstractmethod
    def visit_block(self, node: List[ir.StmtBase]):
        raise NotImplementedError("Block rewriter must be subclassed.")

    def visit(self, node: list):
        repl = []
        for block in sequence_blocks(node):
            leading = block[0]
            if isinstance(leading, ir.IfElse):
                if_branch = self.visit(leading.if_branch)
                else_branch = self.visit(leading.else_branch)
                rewrite = ir.IfElse(leading.test, if_branch, else_branch, leading.pos)
                repl.append(rewrite)
            elif isinstance(leading, ir.ForLoop):
                body = self.visit(leading.body)
                rewrite = ir.ForLoop(leading.target, leading.iterable, body, leading.pos)
                repl.append(rewrite)
            elif isinstance(leading, ir.WhileLoop):
                body = self.visit(leading.body)
                rewrite = ir.WhileLoop(leading.test, body, leading.pos)
                repl.append(rewrite)
            else:
                rewrite = self.visit_block(block)
                repl.extend(rewrite)
        return repl
