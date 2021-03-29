import itertools
import operator
import typing
from functools import singledispatch, singledispatchmethod

import ir
from utils import negate_condition
from visitor import TransformBase, walk_branches


class TerminatedPath(ir.StmtBase):
    """
    sentinel providing an unambiguous method to convey that an unterminated path
    is not available
    """
    __slots__ = ()


def unreachable_branches(node: ir.IfElse):
    """
    Determines whether either branch can statically be determined to be unreachable in simple
    cases.
    """
    if not node.test.constant:
        return False, False
    if isinstance(node.test, ir.Expression):
        raise NotImplementedError("constant expression folding isn't yet implemented")
    branch_cond = operator.truth(node.test)
    return (False, True) if operator.truth(branch_cond) else (True, False)


def simplify_branch(node: ir.IfElse) -> typing.Union[ir.IfElse, list]:
    unreachable = unreachable_branches(node)
    if unreachable == (True, False):
        node = node.else_branch
    elif unreachable == (False, True):
        node = node.if_branch
    else:
        if not node.if_branch:
            test = negate_condition(node.test)
            if_branch, else_branch = node.else_branch, node.if_branch
            pos = node.pos
            node = ir.IfElse(test, if_branch, else_branch, pos)
    return node


def contains_break(entry):
    for stmt in walk_branches(entry):
        if isinstance(stmt, ir.Break):
            return True
    return False


@singledispatch
def terminates_control_flow(node):
    msg = f"No method to test whether control flow may pass through node of type {type(node)}"
    raise TypeError(msg)


@terminates_control_flow.register
def _(node: ir.StmtBase):
    return node.is_terminator


@terminates_control_flow.register
def _(node: list):
    return terminates_control_flow(node[-1]) if node else False


@terminates_control_flow.register
def _(node: ir.IfElse):
    unreachable = unreachable_branches(node)
    if unreachable == (True, False):
        return terminates_control_flow(node.else_branch)
    elif unreachable == (False, True):
        return terminates_control_flow(node.if_branch)
    else:
        return terminates_control_flow(node.if_branch) and terminates_control_flow(node.else_branch)


def terminated_branches(node: ir.IfElse) -> typing.Tuple[bool, bool]:
    unreachable = unreachable_branches(node)
    if unreachable == (True, False):
        return True, terminates_control_flow(node.else_branch)
    elif unreachable == (False, True):
        return terminates_control_flow(node.if_branch), False
    else:
        return terminates_control_flow(node.if_branch), terminates_control_flow(node.else_branch)


def find_unterminated_stmtlist(stmt_list):
    if not isinstance(stmt_list, list):
        raise TypeError("expected a list of statements")
    if stmt_list:
        last_stmt = stmt_list[-1]
        if isinstance(last_stmt, ir.IfElse):
            terminated = terminated_branches(last_stmt)
            if terminated == (True, True):
                return TerminatedPath()
            elif terminated == (True, False):
                return find_unterminated_path(last_stmt.else_branch)
            elif terminated == (False, True):
                return find_unterminated_path(last_stmt.if_branch)
        elif terminates_control_flow(last_stmt):
            return TerminatedPath()
    return stmt_list


def get_unused_name(gen=itertools.count()):
    return ir.NameRef(f"tmp_{next(gen)}")


def find_unterminated_path(stmts: list):
    if len(stmts) == 0:
        # empty statement list
        return stmts
    last_stmt = stmts[-1]
    if isinstance(last_stmt, ir.IfElse):
        # Check for reachable branches
        normalized = simplify_branch(last_stmt)
        if not isinstance(normalized, ir.IfElse):
            return find_unterminated_path(normalized)
        if_path = find_unterminated_path(normalized.if_branch)
        else_path = find_unterminated_path(normalized.else_branch)
        if isinstance(if_path, TerminatedPath):
            return else_path
        elif isinstance(else_path, TerminatedPath):
            return if_path
    elif terminates_control_flow(last_stmt):
        return TerminatedPath()
    return stmts


def remove_trailing_continues(node: typing.Union[list, ir.IfElse]) -> typing.Union[list, ir.IfElse]:
    """
    Remove continues that are the last statement along some execution path without entering
    nested loops.

    """
    if isinstance(node, ir.IfElse):
        true_branch = remove_trailing_continues(node.if_branch)
        false_branch = remove_trailing_continues(node.else_branch)
        return ir.IfElse(node.test, true_branch, false_branch, node.pos)
    elif node:
        last_stmt = node[-1]
        if isinstance(last_stmt, ir.Continue):
            node = node[:-1]
        elif isinstance(last_stmt, ir.IfElse):
            last_stmt = remove_trailing_continues(last_stmt)
            node = node[:-1]
            node.append(last_stmt)
    return node


class RemoveContinues(TransformBase):

    def __call__(self, entry):
        return self.visit(entry)

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: list):
        repl = []
        for stmt in node:
            repl.append(self.visit(stmt))
        return repl

    @visit.register
    def _(self, node: ir.ForLoop):
        repl = self.visit(node.body)
        repl = remove_trailing_continues(repl)
        # Fix branches that now look like
        # if cond:
        #   blank...
        # else:
        #   do_something
        for index, stmt in enumerate(repl):
            if isinstance(stmt, ir.IfElse):
                repl[index] = simplify_branch(stmt)
        return ir.ForLoop(node.assigns, repl, node.pos)

    @visit.register
    def _(self, node: ir.WhileLoop):
        repl = self.visit(node.body)
        repl = remove_trailing_continues(repl)
        return ir.WhileLoop(node.test, repl, node.pos)


class MergePaths(TransformBase):
    """

    At any branch point, corresponding to an if else condition, where one branch is terminated by break, continue,
    or return move any statements that follow the ifelse to the unterminated path.

    This ensures that whenever we conditionally encounter a break, return, or continue within
    a loop, it's encountered as the last statement along that path within the loop.


    """

    def __init__(self):
        self.loops = []

    def __call__(self, node):
        if self.loops:
            raise RuntimeError("Cannot enter visitor with existing loops")
        node = self.visit(node)
        if self.loops:
            raise RuntimeError("Ending in an inconsistent loop state")
        return node

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.ForLoop):
        self.loops.append(node)
        assigns = node.assigns
        body = self.visit(node.body)
        pos = node.pos
        header = self.loops.pop()
        assert (header is node)
        return ir.ForLoop(assigns, body, pos)

    @visit.register
    def _(self, node: ir.WhileLoop):
        test = node.test
        if test.constant:
            if not operator.truth(test):
                return []
        self.loops.append(node)
        body = self.visit(node.body)
        pos = node.pos
        header = self.loops.pop()
        assert (header is node)
        return ir.WhileLoop(test, body, pos)

    @visit.register
    def _(self, node: ir.IfElse):
        node = simplify_branch(node)
        if isinstance(node, ir.IfElse):
            if_branch = self.visit(node.if_branch)
            else_branch = self.visit(node.else_branch)
            node = ir.IfElse(node.test, if_branch, else_branch, node.pos)
        return node

    @visit.register
    def _(self, node: list):
        repl = []
        append_to = repl
        for stmt in node:
            stmt = self.visit(stmt)
            if isinstance(stmt, list):
                repl.extend(stmt)
            else:
                append_to.append(stmt)
                if isinstance(stmt, ir.IfElse):
                    terminated = terminated_branches(stmt)
                    if terminated == (True, False):
                        append_to = find_unterminated_path(stmt.else_branch)
                    elif terminated == (False, True):
                        append_to = find_unterminated_path(stmt.if_branch)
                    elif terminated == (True, True):
                        break
                elif terminates_control_flow(stmt):
                    break
        return repl
