from collections import defaultdict
from functools import singledispatchmethod
from itertools import chain, islice
from typing import Dict, Iterator

import ir

from errors import CompilerError
from reductions import ExpressionMapper
from symbol_table import symbol_table
from visitor import StmtVisitor, walk

# Need can flatten check
# Need check for live escapes

# Flatten should run first, then optimization of resulting ternary ops
# Two sided memory ops with same ordering should have writes combined

# check for simple accumulate branch like
#  for i in range(n):
#      if a[i] < b[i]:
#          a[i] += b[i]


# mirrored memory ordering
# needs npyv_xor_ extensions


#  for i in range(n):
#      if a[i] < b[i]:
#          a[i] += b[i]
#      else:
#          a[i] -= b[i]

# Check for reducible to noop
# eg.
# while cond(v):
#    v *= ...

# Here multiplication, division, addition, subtraction, simple assign can be reduced to identity

# Check that either write and read arrays don't overlap or write patterns track and follow read patterns

# Check for reducible to identity

# if varying_cond:
#     a += v
# else:
#     pass

# Here we would want to rename first, so
# a0 = a + v
# a = a0 if varying_cond else a

# So here we have either expression or identity
# looking back at expression, we would need to be able to collapse
# this is probably going to be a different pass, if it's only used conditionally


def is_terminal(stmt):
    return isinstance(stmt, (ir.IfElse, ir.ForLoop, ir.WhileLoop, ir.Break, ir.Continue, ir.Return))


def is_entry_point(stmt: ir.StmtBase):
    return isinstance(stmt, (ir.IfElse, ir.ForLoop, ir.WhileLoop))


def is_optimizable_block(stmts: list, start: int):
    return not is_entry_point(stmts[start])


def is_exit_point(stmt: ir.StmtBase):
    """
    check if it exits a control flow region
    :param stmt:
    :return:
    """
    return isinstance(stmt, (ir.Break, ir.Return, ir.Continue))


def collect_local_assignments(stmts: list, start: int, stop: int):
    assigns = defaultdict(set)
    for stmt in islice(stmts, start, stop):
        # break at end of any basic block, ignoring call sites
        if is_exit_point(stmt):
            return assigns
        assigns[stmt.target].add(stmt.value)
    return assigns


def gather_local_ternary(stmts: list, start: int, stop: int):
    ternary_ops = set()
    for stmt in islice(stmts, start, stop):
        # break at end of any basic block, ignoring call sites
        if is_exit_point(stmt):
            return ternary_ops
        if isinstance(stmt, ir.Assign):
            ternary_ops.update(e for e in chain(walk(stmt.target), walk(stmt.value)) if isinstance(e, ir.Select))
        elif isinstance(stmt, ir.InPlaceOp):
            ternary_ops.update(e for e in chain(walk(stmt.target), walk(stmt.value)) if isinstance(e, ir.Select))
        elif isinstance(stmt, ir.SingleExpr):
            ternary_ops.update(e for e in walk(stmt.expr) if isinstance(e, ir.Select))
    return ternary_ops


def reconstruct_expr(node: ir.Expression, rewrites: Dict[ir.ValueRef, ir.ValueRef]):
    for value in walk(node):
        if isinstance(value, ir.Expression):
            repl = rewrites.get(value)
            if repl is None:
                repl = value.reconstruct(*(rewrites.get(subexpr, subexpr) for subexpr in value.subexprs))
                # map to self if identical
                rewrites = value if repl == value else repl
    return rewrites[node]


def collect_control_flow_blocks(stmts: list):
    """
    This gathers control flow blocks from a single statement list.
    It ignores things like branch and loop statements.

    :param stmts:
    :return:
    """
    if not stmts:
        return []
    blocks = []
    segment_begin = None
    for index, stmt in stmts:
        # assume that unreachable code due to continue, break, return is removed
        is_control_flow_entry = is_entry_point(stmt)
        if is_control_flow_entry:
            if segment_begin is not None:
                # terminate existing segment
                blocks.append((segment_begin, index))
                segment_begin = None
            blocks.append((index, index + 1))
        elif segment_begin is None:
            segment_begin = index
    if segment_begin is not None:
        blocks.append((segment_begin, len(stmts)))
    return blocks


def check_no_trivially_unreachable(stmts: list, start: int, stop: int):
    assert stop > start
    if stop - start > 1:
        for stmt in islice(reversed(stmts), start+1, stop):
            if is_exit_point(stmt):
                msg = f"Trivially dead code found due to unexpected exit statement {stmt}."
                raise CompilerError(msg)


def unpack_assignment(node: ir.StmtBase):
    if isinstance(node, (ir.Assign, ir.InPlaceOp)):
        return node.target, node.value
    else:
        return None


def is_assignment(node: ir.StmtBase):
    return isinstance(node, (ir.Assign, ir.InPlaceOp))


def assign_iter(block: Iterator[ir.StmtBase]):
    for stmt in block:
        if isinstance(stmt, (ir.Assign, ir.InPlaceOp)):
            in_place = isinstance(stmt, ir.InPlaceOp)
            yield stmt.target, stmt.value, in_place, stmt.pos


def block_iter(stmts: list):
    for start, stop in collect_control_flow_blocks(stmts):
        yield islice(stmts, start, stop)


def map_expression(node: ir.ValueRef, assigned: Dict[ir.ValueRef, ir.ValueRef]):
    if isinstance(node, ir.Expression):
        subexprs = []
        for subexpr in node.subexprs:
            subexprs.append(map_expression(subexpr, assigned))
        node = node.reconstruct(*subexprs)
    return assigned.get(node, node)


def eliminate_common_subexprs(stmts: list, symbols: symbol_table):
    mapper = ExpressionMapper(symbols)
    for block in block_iter(stmts):
        for stmt in block:
            if is_assignment(stmt):
                target = stmt.target
                value = stmt.value
                if isinstance(target, ir.NameRef):
                    pass
                # mapper.map_expr()


class SimpleBranchFlattenCheck(StmtVisitor):

    def __init__(self):
        pass

    def __call__(self, node: ir.IfElse):
        pass

    @singledispatchmethod
    def visit(self, node):
        raise NotImplementedError

    @visit.register
    def _(self, node: ir.Assign):
        return not isinstance(node.target, ir.Subscript)

    @visit.register
    def _(self, node: ir.InPlaceOp):
        return not isinstance(node.target, ir.Subscript)

    @visit.register
    def _(self, node: list):
        return all(self.visit(stmt) for stmt in node)

    @visit.register
    def _(self, node: ir.IfElse):
        return self.visit(node.if_branch) and self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.SingleExpr):
        return not isinstance(node.expr, ir.Call)

    @visit.register
    def _(self, node: ir.WhileLoop):
        pass

    @visit.register
    def _(self, node: ir.ForLoop):
        return False
