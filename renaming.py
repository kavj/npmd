from contextlib import contextmanager
from functools import singledispatchmethod
from typing import Dict, Iterable, Optional

import ir

from reductions import ExpressionMapper
from utils import unpack_iterated, contains_stmt_types
from symbol_table import symbol_table
from type_checks import TypeHelper
from visitor import walk, StmtVisitor, StmtTransformer


def is_control_flow(node):
    return isinstance(node, (ir.IfElse, ir.ForLoop, ir.WhileLoop, ir.Break, ir.Continue))


"""
Jotting down the required components...


Contiguous load check

continuous write check

check escaping --- can just run reaching_check

need something to estimate practical limits on unrolling 

"""


def rewrite_expression(node: ir.ValueRef, bound: Dict[ir.ValueRef, ir.ValueRef]):
    local_defs = {}
    for term in walk(node):
        if isinstance(term, ir.Expression):
            repl_term = local_defs.get(term)
            if repl_term is None:
                repl_term = term.reconstruct(*(local_defs.get(subexpr) for subexpr in term.subexprs))
        else:
            repl_term = bound.get(term, term)
        local_defs[term] = repl_term
    return local_defs.get(node)


class LoopLocalValueNumbering(StmtVisitor):
    """
    Value numbering should track current def per variable and names bound to any expression.

    latest indicates remapping of variable names
    expression mapper stores anything explicitly named

    anything that leaves loop must be copied out

    good for dealing with branch flattening


    """

    def __init__(self, mapper: ExpressionMapper, latest: Optional[Dict]=None):
        self.mapper = mapper
        self.latest = latest if latest is not None else {}

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Assign):
        value = rewrite_expression(node.value, self.latest)
        name = self.mapper.map_expr(value)
        if isinstance(node.target, ir.NameRef):
            # rename and bind the latest def to renamed
            self.latest[node.target] = name
            target = name
        else:
            target = node.target
        stmt = ir.Assign(target, value, node.pos)
        return stmt

    @visit.register
    def _(self, node: ir.IfElse):
        # These are tricky. We have to check branch forms.
        # In all cases, ignore anything with loop inside a branch.
        # we need to test for several cases
        # 1. neither side contains subscripted writes
        # 2. only one branch is non-empty
        # 3. both branches contain the same subscript targets, appearing at the end of each branch

        # After that there's the issue of whether we have uniform value out, reducible to min max
        # or something that requires a blend/select op.

        # Lastly there's the issue of break statements..

        # typically we start this if there are non-uniform branch conditions
        test = rewrite_expression(node.test, self.latest)
        # copy state
        latest_if = self.latest.copy()
        if_branch = LoopLocalValueNumbering(self.mapper, latest_if).visit(node.if_branch)
        latest_else = self.latest.copy()
        else_branch = LoopLocalValueNumbering(self.mapper, latest_else).visit(node.else_branch)

    
