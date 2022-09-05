import itertools

import npmd.ir as ir

from npmd.errors import CompilerError
from npmd.pretty_printing import PrettyFormatter
from npmd.traversal import get_statement_lists
from npmd.utils import extract_name


def stmt_matches(a: ir.StmtBase, b: ir.StmtBase):
    if isinstance(a, ir.Break) and isinstance(b, ir.Break):
        return True
    elif isinstance(a, ir.Continue) and isinstance(b, ir.Continue):
        return True
    elif isinstance(a, ir.Return) and isinstance(b, ir.Return):
        return a.value == b.value


def hoist_control_flow(node: ir.Function):
    """

    This moves control flow statements that appear at the beginning or end of both sides of a branch
    out of that branch.

    :param node:
    :return:
    """

    for stmt_list in get_statement_lists(node):
        if any(isinstance(stmt, ir.IfElse) for stmt in stmt_list):
            hoisted = False
            repl = []
            for stmt in stmt_list:
                if isinstance(stmt, ir.IfElse):
                    if_branch = stmt.if_branch
                    else_branch = stmt.else_branch

                    while if_branch and else_branch:
                        leading = if_branch[0]
                        if isinstance(leading, (ir.Break, ir.Continue, ir.Return)):
                            if stmt_matches(leading, else_branch[0]):
                                repl.append(leading)
                                if_branch.pop(0)
                                else_branch.pop(0)
                                hoisted = True
                                continue
                        break

                    repl.append(stmt)

                    while if_branch and else_branch:
                        leading = else_branch[-1]
                        if isinstance(leading, (ir.Break, ir.Continue, ir.Return)):
                            if stmt_matches(leading, if_branch[-1]):
                                repl.append(leading)
                                if_branch.pop()
                                else_branch.pop()
                                hoisted = True
                                continue
                        break
                else:
                    repl.append(stmt)

            if hoisted:
                # if we hoisted anything from this branch, we have to transfer it.
                # this won't hoist nested branches or loops. It's mostly to extract
                # things like common break and continue statements, so that the CFG
                # doesn't have to deal the resulting divergence.
                stmt_list.clear()
                stmt_list.extend(repl)
            repl.clear()


def split_common_branch_statements(node: ir.Function):
    """
    splits out statements that are part of a prefix or suffix common to all branches of the branch statement.
    :param node:
    :return:
    """
    for stmt_list in get_statement_lists(node):
        hoisted = False
        repl = []
        if any(isinstance(stmt, ir.IfElse) for stmt in stmt_list):
            for stmt in stmt_list:
                if isinstance(stmt, ir.IfElse):
                    # first look for a shared prefix
                    if_branch = stmt.if_branch
                    else_branch = stmt.else_branch

                    while if_branch and else_branch:
                        if stmt_matches(if_branch[0], else_branch[0]):
                            hoisted = True
                            common = if_branch.pop(0)
                            repl.append(common)
                            else_branch.pop(0)
                        else:
                            break

                    # now append the rest of the branch
                    repl.append(stmt)

                    # now look for a remaining common suffix
                    while if_branch and else_branch:
                        if stmt_matches(if_branch[-1], else_branch[-1]):
                            hoisted = True
                            common = else_branch.pop()
                            repl.append(common)
                            if_branch.pop()
                        else:
                            break

        if hoisted:
            # if we hoisted anything from this branch, we have to transfer it.
            # this won't hoist nested branches or loops. It's mostly to extract
            # things like common break and continue statements, so that the CFG
            # doesn't have to deal the resulting divergence.
            stmt_list.clear()
            stmt_list.extend(repl)
            repl.clear()


def add_trivial_return(node: ir.Function):
    """
    Method to add a trivial return to function.
    This will typically get cleaned up later if it's not reachable.
    :param node:
    :return:
    """

    if len(node.body) > 0:
        if isinstance(node.body[-1], ir.Return):
            return
        last = node.body[-1]
        pos = ir.Position(last.pos.line_end + 1, last.pos.line_end + 1, 1, 40)
        node.body.append(ir.Return(ir.NoneRef(), pos))
    else:
        pos = ir.Position(-1, -1, 1, 40)
        node.body.append(ir.Return(ir.NoneRef(), pos))


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
        assert len(node.args) == 1
        node = ir.SingleDimRef(node.args[0], ir.Zero)
    elif func_name == 'min':
        assert len(node.args) > 1
        terms = ir.MIN(node.args[0], node.args[1])
        for arg in itertools.islice(node.args, 2, None):
            terms = ir.MIN(terms, arg)
        node = terms
    elif func_name == 'max':
        assert len(node.args) > 1
        terms = ir.MAX(node.args[0], node.args[1])
        for arg in itertools.islice(node.args, 2, None):
            terms = ir.MAX(terms, arg)
        node = terms
    return node
