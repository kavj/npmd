from typing import List

import ir

from errors import CompilerError
from pretty_printing import PrettyFormatter

from utils import extract_name


def replace_call(node: ir.Call):
    func_name = extract_name(node)
    if func_name == "numpy.ones":
        node = ir.ArrayInitializer(*node.args, ir.One)
    elif func_name == "numpy.zeros":
        node = ir.ArrayInitializer(*node.args, ir.Zero)
    elif func_name == "numpy.empty":
        node = ir.ArrayInitializer(*node.args, ir.NoneRef())
    elif func_name == "zip":
        node = ir.Zip(*node.args)
    elif func_name == "enumerate":
        node = ir.Enumerate(*node.args)
    elif func_name == "range":
        nargs = len(node.args)
        if nargs == 3:
            node = ir.AffineSeq(*node.args)
        elif nargs == 2:
            start, stop = node.args
            node = ir.AffineSeq(start, stop, ir.One)
        elif nargs == 1:
            stop, = node.args
            node = ir.AffineSeq(ir.Zero, stop, ir.One)
        else:
            pf = PrettyFormatter()
            msg = f"bad arg count for call to range {pf(node)}"
            raise CompilerError(msg)
    elif func_name == "len":
        node = ir.Length(*node.args)
    return node


def flatten_branch(node: ir.IfElse):
    """
    :param node:
    :return:
    """

    flattenable = []
    branch = node

    while len(branch.else_branch) == 1 and isinstance(branch.else_branch[0], ir.IfElse):
        flattenable.append(branch)
        branch = branch.else_branch[0]

    if not flattenable:
        return node

    conditions = [b.test for b in flattenable]
    conditions.append(branch.test)
    if_branches = [b.if_branch for b in flattenable]
    if_branches.append(branch.if_branch)
    default_ = branch.else_branch
    return ir.Case(conditions, if_branches, default_)


def rewrite_branches(node: List[ir.StmtBase]):
    """
    Converts nested branches to cases.
    :param node:
    :return:
    """
    if isinstance(node, ir.Function):
        body = rewrite_branches(node.body)
        return ir.Function(node.name, node.args, body)
    repl = []
    for stmt in node:
        if isinstance(stmt, ir.IfElse):
            stmt = flatten_branch(stmt)
            if isinstance(stmt, ir.Case):
                branches = [rewrite_branches(b) for b in stmt.branches]
                stmt.branches = branches
                repl.append(stmt)
            else:
                assert isinstance(stmt, ir.IfElse)
                if_branch = rewrite_branches(stmt.if_branch)
                else_branch = rewrite_branches(stmt.else_branch)
                stmt = ir.IfElse(stmt.test, if_branch, else_branch, stmt.pos)
                repl.append(stmt)
        elif isinstance(stmt, ir.ForLoop):
            body = rewrite_branches(stmt.body)
            stmt = ir.ForLoop(stmt.target, stmt.iterable, body, stmt.pos)
            repl.append(stmt)
        elif isinstance(stmt, ir.WhileLoop):
            body = rewrite_branches(stmt.body)
            stmt = ir.WhileLoop(stmt.test, body, stmt.pos)
            repl.append(stmt)
        else:
            repl.append(stmt)
    return repl
