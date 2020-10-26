import itertools
import operator
from functools import singledispatch, singledispatchmethod

import ir
from utils import negate_condition
from visitor import TransformBase


@singledispatch
def is_terminated(node):
    msg = f"Termination check is not implemented for {type(node)}"
    raise NotImplementedError(msg)


@is_terminated.register
def _(node: ir.IfElse):
    if node.test.is_constant:
        t = operator.truth(node.test)
        if t:
            return is_terminated(node.if_branch)
        else:
            return is_terminated(node.else_branch)
    return is_terminated(node.if_branch) and is_terminated(node.else_branch)


@is_terminated.register
def _(node: ir.StmtBase):
    return node.is_terminator


@is_terminated.register
def _(node: list):
    for stmt in node:
        if stmt.is_terminator:
            return True
        elif stmt.is_conditional_branch:
            if is_terminated(stmt):
                return True
    return False


def find_unterminated_stmtlist(node):
    if not isinstance(node, list):
        raise TypeError("expected a list of statements")
    if node:
        if node[-1].is_terminator:
            return
        elif node[-1].is_conditional_branch:
            if_branch = find_unterminated_stmtlist(node[-1].if_branch)
            else_branch = find_unterminated_stmtlist(node[-1].else_branch)
            if if_branch is None:
                if else_branch is None:
                    return
                else:
                    return else_branch
            elif else_branch is None:
                return if_branch
            else:  # ends in a pair of unterminated branches
                return node
        else:  # unterminated statement list
            return node
    else:  # empty unterminated statement list
        return node


class RemoveUnreachable(TransformBase):
    """
    Main point of unreachable code removal.
    This removes statements that are statically blocked by {Continue, Break, Return}.
    This also removes identifiable dead branches.

    # Example 1

    if True:
        ...
    else:
        ...  # unreachable, since condition is never False, we can merge the True branch statements into the parent list

    
    # Example 2

    if condition:
        ...
        continue
    else:
        continue

    ... # unreachable, since both branches terminate control flow

    """

    def __init__(self):
        self.loops = []

    def __call__(self, node):
        if self.loops:
            raise RuntimeError("Transformer called with existing loops")
        node = self.visit(node)
        return node

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.IfElse):
        test = node.test
        if_branch = node.if_branch
        else_branch = node.else_branch
        pos = node.pos
        if test.is_constant:
            if test:
                node = ir.IfElse(test, self.visit(if_branch), [], pos)
            else:
                node = ir.IfElse(test, [], self.visit(else_branch), pos)
        else:
            node = ir.IfElse(test, self.visit(if_branch), self.visit(else_branch), pos)
        return node

    @visit.register
    def _(self, node: ir.ForLoop):
        self.loops.append(node)
        node.body = self.visit(node.body)
        header = self.loops.pop()
        assert (header is node)
        return node

    @visit.register
    def _(self, node: ir.WhileLoop):
        self.loops.append(node)
        node.body = self.visit(node.body)
        header = self.loops.pop()
        assert (header is node)
        return node

    @visit.register
    def _(self, stmts: list):
        repl = []
        for s in stmts:
            s = self.visit(s)
            repl.append(s)
            if is_terminated(s):
                break
        return repl


class BranchCleanup(TransformBase):
    """
    Covers removal of now empty ifelse statements
    and normalizes branch conditions so that if either
    branch is non-empty, the true branch is non-empty.


    """

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.Function):
        body = self.visit(node.body)
        node = ir.Function(node.name, node.args, body, node.types, node.arrays)
        return node

    @visit.register
    def _(self, node: ir.IfElse):
        if node.test.is_constant:
            node = self.visit(node.if_branch) if node.test else self.visit(node.else_branch)
        else:
            if_branch = self.visit(node.if_branch)
            else_branch = self.visit(node.else_branch)
            test = node.test
            pos = node.pos
            if if_branch is None:
                if else_branch is None:
                    node = None
                else:
                    test = negate_condition(test)
                    node = ir.IfElse(test, else_branch, [], pos)
            elif else_branch is None:
                # If both are non-empty and one is terminated, move it to the else branch
                node = ir.IfElse(test, if_branch, [], pos)
            elif is_terminated(if_branch) and not is_terminated(else_branch):
                if else_branch:
                    test = negate_condition(test)
                    node = ir.IfElse(test, else_branch, if_branch, pos)
        return node

    @visit.register
    def _(self, node: list):
        repl = []
        for index, stmt in enumerate(node):
            stmt = self.visit(stmt)
            if stmt is not None:
                if isinstance(stmt, list):
                    repl.extend(stmt)
                else:
                    repl.append(stmt)
        return repl


def get_unused_name(gen=itertools.count()):
    return ir.NameRef(f"tmp_{next(gen)}")


def find_unterminated_path(node):
    for stmt in node:
        if stmt.is_terminator:
            return
        elif stmt.is_conditional_branch:
            if_branch_result = find_unterminated_path(stmt.if_branch)
            else_branch_result = find_unterminated_path(stmt.else_branch)
            if if_branch_result is None and else_branch_result is None:
                return
    return node


class RemoveTrailingContinues(TransformBase):
    """
    Removes continues that are the last executable 
    statement in a loop, with respect to any path that contains them

    for v in value:
        ...
        if cond:
            continue

    after this pass, we should have

    for v in value:
        ...
        if cond:

    
    The if statement is now redundant and could be removed as well. 
    
    """

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: list):
        repl = []
        for stmt in node:
            if stmt.is_terminator:
                if not isinstance(stmt, ir.Continue):
                    repl.append(stmt)
                break
            else:
                repl.append(self.visit(stmt))
        return repl

    @visit.register
    def _(self, node: ir.IfElse):
        test = node.test
        if_branch = self.visit(node.if_branch)
        else_branch = self.visit(node.else_branch)
        pos = node.pos
        if else_branch and not if_branch:
            test = negate_condition(test)
            node = ir.IfElse(test, else_branch, if_branch, pos)
        else:
            node = ir.IfElse(test, if_branch, else_branch, pos)
        return node

    @visit.register
    def _(self, node: ir.ForLoop):
        node.body = self.visit(node.body)
        return node


class MergePaths(TransformBase):
    """

    At any branch point, corresponding to an if else condition, where one branch is terminated by break, continue,
    or return move any statements that follow the ifelse to the unterminated path.
    This results in a CFG with fewer stray edges.

    This ensures that whenever we conditionally encounter a break, return, or continue within
    a loop, it's encountered as the last statement along that path within the loop.

    Whenever a continue statement is encountered along one branch of a path

    Starting from something of the form:

    if cond:
        ...
    else:
        ...
        continue

    ... # more statements


    We end with something of the form:

    if cond:
        ...
        ... # more statements
    else:
        continue

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
        target = node.target
        iterable = node.iterable
        body = self.visit(node.body)
        pos = node.pos
        header = self.loops.pop()
        assert (header is node)
        return ir.ForLoop(iterable, target, body, pos)

    @visit.register
    def _(self, node: ir.WhileLoop):
        self.loops.append(node)
        test = node.test
        body = self.visit(node.body)
        pos = node.pos
        header = self.loops.pop()
        assert (header is node)
        return ir.WhileLoop(test, body, pos)

    @visit.register
    def _(self, node: list):
        repl = []
        appendto = repl
        for stmt in node:
            stmt = self.visit(stmt)
            appendto.append(stmt)
            if stmt.is_terminator:
                break
            elif stmt.is_conditional_branch:
                if not stmt.empty:
                    if_branch = find_unterminated_path(stmt.if_branch)
                    else_branch = find_unterminated_path(stmt.else_branch)
                    if if_branch == else_branch is None:
                        break
                    elif if_branch is None:
                        appendto = else_branch
                    elif else_branch is None:
                        appendto = if_branch
        return repl
