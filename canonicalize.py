import numpy
import operator

from functools import singledispatchmethod
from pretty_printing import pretty_formatter

import ir
from errors import CompilerError
from utils import extract_name
from visitor import StmtTransformer


def type_from_numpy_type(t: type):
    if t == numpy.int32:
        return ir.Int32
    elif t == numpy.int64:
        return ir.Int64
    elif t == numpy.bool_:
        return ir.BoolType
    elif t == bool:
        return ir.BoolType
    elif t == numpy.float32:
        return ir.Float32
    elif t == numpy.float64:
        return ir.Float64
    elif t in (ir.Int32, ir.Int64, ir.Float32, ir.Float64):
        return t
    else:
        msg = f"{t} is not a currently supported type."
        raise CompilerError(msg)


def find_unterminated_path(stmts):
    if not isinstance(stmts, list):
        raise TypeError("Internal Error: expected a list of statements")
    if len(stmts) > 0:
        last = stmts[-1]
        if isinstance(last, (ir.Continue, ir.Break, ir.Return)):
            return
        elif isinstance(last, ir.IfElse):
            if last.test.constant:
                # If we have a constant branch condition,we can only follow
                # the reachable branch
                if operator.truth(last.test):
                    return find_unterminated_path(last.if_branch)
                else:
                    return find_unterminated_path(last.else_branch)
            else:
                if_path = find_unterminated_path(last.if_branch)
                else_path = find_unterminated_path(last.else_branch)
                if if_path is None and else_path is None:
                    return  # terminated
                elif if_path is None:
                    return else_path
                elif else_path is None:
                    return if_path
    return stmts


def remove_trailing_continues(node: list) -> list:
    """
    Remove continues that are the last statement along some execution path within the current
    enclosing loop.

    """

    if len(node) > 0:
        last = node[-1]
        if isinstance(last, ir.IfElse):
            if_branch = remove_trailing_continues(last.if_branch)
            else_branch = remove_trailing_continues(last.else_branch)
            if if_branch != last.if_branch or else_branch != last.else_branch:
                last = ir.IfElse(last.test, if_branch, else_branch, last.pos)
                # copy original
                node = node[:-1]
                node.append(last)
        elif isinstance(last, ir.Continue):
            node = node[:-1]
    return node


class NormalizePaths(StmtTransformer):
    """
    This is the tree version of control flow optimization.
    It removes any statements blocked by break, return, or continue
    and inlines paths as an alternative to explicit continue statements.

    """

    def __init__(self):
        self.innermost_loop = None
        self.body = None

    def __call__(self, node):
        repl = self.visit(node)
        return repl

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.ForLoop):
        body = self.visit(node.body)
        body = remove_trailing_continues(body)
        if body != node.body:
            node = ir.ForLoop(node.target, node.iterable, body, node.pos)
        return node

    @visit.register
    def _(self, node: ir.WhileLoop):
        test = node.test
        if test.constant:
            if not operator.truth(test):
                # return None if the loop body is unreachable.
                return
            body = self.visit(node.body)
            body = remove_trailing_continues(body)
            if body != node.body:
                node = ir.WhileLoop(node.test, body, node.pos)
        return node

    @visit.register
    def _(self, node: ir.IfElse):
        if_branch = self.visit(node.if_branch)
        else_branch = self.visit(node.else_branch)
        if not (if_branch or else_branch):
            return
        node = ir.IfElse(node.test, if_branch, else_branch, node.pos)
        return node

    @visit.register
    def _(self, node: list):
        repl = []
        append_to = repl
        for stmt in node:
            if isinstance(stmt, ir.IfElse):
                if stmt.test.constant:
                    live_branch = stmt.if_branch if operator.truth(stmt.test) else stmt.else_branch
                    live_branch = self.visit(live_branch)
                    extendable_path = find_unterminated_path(live_branch)
                    append_to.extend(live_branch)
                    if extendable_path is not live_branch:
                        if extendable_path is None:
                            break
                        else:  # extendable path exists and is distinct from the inlined branch
                            append_to = extendable_path
                else:
                    stmt = self.visit(stmt)
                    if stmt is None:
                        continue  # doesn't execute anything
                    append_to.append(stmt)
                    if_path = find_unterminated_path(stmt.if_branch)
                    else_path = find_unterminated_path(stmt.else_branch)
                    if if_path is None and else_path is None:
                        break  # remaining statements are unreachable
                    elif if_path is None:
                        append_to = else_path
                    elif else_path is None:
                        append_to = if_path
            else:
                stmt = self.visit(stmt)
                if stmt is not None:
                    append_to.append(stmt)
                    if isinstance(stmt, (ir.Break, ir.Continue, ir.Return)):
                        break  # remaining statements are unreachable
        return repl


def replace_call(node: ir.Call):
    func_name = extract_name(node)
    if func_name == "numpy.ones":
        node = ir.Ones(*node.args)
    elif func_name == "numpy.zeros":
        node = ir.Zeros(*node.args)
    elif func_name == "numpy.empty":
        node = ir.Empty(*node.args)
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
            pf = pretty_formatter()
            msg = f"bad arg count for call to range {pf(node)}"
            raise CompilerError(msg)
    elif func_name == "len":
        node = ir.Length(*node.args)
    return node
