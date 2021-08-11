import operator
import typing
from contextlib import contextmanager
from functools import singledispatchmethod

import ir
from errors import CompilerError
from visitor import StmtTransformer


class TerminatedPath(ir.StmtBase):
    """
    sentinel providing an unambiguous method to convey that an unterminated path
    is not available
    """
    __slots__ = ()


def clear_dead_branches(node: ir.IfElse) -> typing.Union[ir.IfElse, list]:
    """
    Remove statements from unreachable branches, but keep the branch intact.
    """
    if node.test.constant:
        if operator.truth(node.test):
            node = ir.IfElse(node.test, node.if_branch, [], node.pos)
        else:
            node = ir.IfElse(node.test, [], node.else_branch, node.pos)
    return node


def contains_break(entry):
    raise NotImplementedError


def find_unterminated_path(stmts):
    if not isinstance(stmts, list):
        raise TypeError("Internal Error: expected a list of statements")
    if len(stmts) > 0:
        last = stmts[-1]
        if isinstance(last, (ir.Continue, ir.Break, ir.Return)):
            return TerminatedPath()
        elif isinstance(last, ir.IfElse):
            if last.test.constant:
                # If we have a constant branch condition,we can only follow
                # the reachable branch
                if operator.truth(last.test):
                    return find_unterminated_path(last.if_branch)
                else:
                    return find_unterminated_path(last.else_branch)
            else:
                if_branch = find_unterminated_path(last.if_branch)
                else_branch = find_unterminated_path(last.else_branch)
                if_br_terminated = isinstance(if_branch, TerminatedPath)
                else_br_terminated = isinstance(else_branch, TerminatedPath)
                if if_br_terminated:
                    if else_br_terminated:
                        return TerminatedPath()
                    else:
                        return else_branch
                elif else_br_terminated:
                    return if_branch
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

    @property
    def within_loop(self):
        return self.innermost_loop is not None

    @contextmanager
    def enclosing_loop(self, header):
        stashed = self.innermost_loop
        self.innermost_loop = header
        yield
        assert self.innermost_loop is header
        self.innermost_loop = stashed

    def __call__(self, node):
        if self.within_loop:
            msg = "Internal Error: Entering visitor from inconsistent state."
            raise RuntimeError(msg)
        repl = self.visit(node)
        if self.within_loop:
            msg = "Internal Error: Exiting visitor from inconsistent state."
            raise RuntimeError(msg)
        return repl

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.ForLoop):
        with self.enclosing_loop(node):
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
                return
        with self.enclosing_loop(node):
            body = self.visit(node.body)
            body = remove_trailing_continues(body)
            if body != node.body:
                node = ir.WhileLoop(node.test, body, node.pos)
        return node

    @visit.register
    def _(self, node: ir.IfElse):
        node = clear_dead_branches(node)
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
            if isinstance(stmt, ir.IfElse):
                if stmt.test.constant:
                    if operator.truth(stmt.test):
                        live_branch = self.visit(stmt.if_branch)
                    else:
                        live_branch = self.visit(stmt.else_branch)
                    live_path = find_unterminated_path(live_branch)
                    append_to.extend(live_branch)
                    if isinstance(live_path, TerminatedPath):
                        break  # any remaining statements are unreachable
                    append_to = live_path
                else:
                    if_branch = self.visit(stmt.if_branch)
                    else_branch = self.visit(stmt.else_branch)
                    stmt = ir.IfElse(stmt.test, if_branch, else_branch, stmt.pos)
                    append_to.append(stmt)
                    if_path = find_unterminated_path(stmt.if_branch)
                    else_path = find_unterminated_path(stmt.else_branch)
                    if_terminated = isinstance(if_path, TerminatedPath)
                    else_terminated = isinstance(else_path, TerminatedPath)
                    if if_terminated and else_terminated:
                        break  # any remaining statements are unreachable
                    elif if_terminated:
                        append_to = else_path
                    elif else_terminated:
                        append_to = if_path
            else:
                stmt = self.visit(stmt)
                if stmt is not None:
                    append_to.append(stmt)
                if isinstance(stmt, (ir.Break, ir.Continue)):
                    if not self.within_loop:
                        stmt_str = "Break" if isinstance(stmt, ir.Break) else "Continue"
                        msg = f"{stmt_str} statement encountered outside of loop, line: {stmt.pos.line_begin}."
                        raise CompilerError(msg)
                    break  # remaining statements are unreachable
                elif isinstance(stmt, ir.Return):
                    if self.within_loop:
                        msg = f"Return statements within a for or while loop are not supported, " \
                              f"line: {stmt.pos.line_begin}."
                        raise CompilerError(msg)
                    break  # remaining statements are unreachable
        return repl


class CallSpecialize:
    """
    Class to set up specialized call replacements. This is meant to be as simple as possible.
    Alternatively, it's possible to write a visitor that performs more involved replacements.

    This is forced to consider default arguments, because some builtins require it.
    It does not support * or ** signatures.

    """

    def __init__(self, name, args, repl, defaults, allows_keywords=False):
        self.name = name
        if len({*args}) < len(args):
            msg = f"Call lowering for {name} has duplicate argument fields, {args} received."
            raise ValueError(msg)
        self.args = args
        self.defaults = defaults
        self.allow_keywords = allows_keywords
        # Check no duplicates
        assert len(self.args) == len(args)
        # Check anything with a default also appears in args
        assert all(arg in self.args for (arg, value) in defaults.items())
        # repl must know how to take a dictionary of arguments and construct
        # a replacement node, suitable for downstream use
        self.replacement = repl

    @property
    def max_arg_count(self):
        return len(self.args)

    @property
    def default_count(self):
        return len(self.defaults)

    @property
    def min_arg_count(self):
        return len(self.args) - len(self.defaults)

    def validate_call(self, args, keywords):
        arg_count = len(args)
        kw_count = len(keywords)
        if not self.allow_keywords and kw_count > 0:
            raise CompilerError(f"Function {self.name} does not allow keyword arguments.")
        mapped = {}
        unrecognized = set()
        duplicates = set()
        missing = set()
        if arg_count + kw_count > self.max_arg_count:
            raise CompilerError(f"Function {self.name} has {self.max_arg_count} fields. "
                                f"{arg_count + kw_count} arguments were provided.")
        for field, arg in zip(self.args, args):
            mapped[field] = arg
        for kw, value in keywords:
            if kw not in self.args:
                unrecognized.add(kw)
            elif kw in mapped:
                duplicates.add(kw)
            mapped[kw] = value
        for field in self.args:
            if field not in mapped:
                if field in self.defaults:
                    mapped[field] = self.defaults[field]
                else:
                    missing.add(field)
        for u, v in unrecognized:
            raise CompilerError(f"unrecognized field {u} in call to {self.name}")
        return self.replacement(mapped)

    def validate_simple_call(self, args):
        # Check call with no keywords
        arg_count = len(args)
        mapped = {}
        if arg_count > self.max_arg_count:
            raise CompilerError(f"Signature for {self.name} accepts {self.max_arg_count} arguments. "
                                f"{arg_count} arguments received.")
        elif arg_count < self.min_arg_count:
            raise CompilerError(f"Signature for {self.name} expects at least {self.min_arg_count} arguments, "
                                f"{arg_count} arguments received.")
        for field, arg in zip(self.args, args):
            mapped[field] = arg
        for field, value in self.defaults.items():
            if field not in mapped:
                mapped[field] = value
        return mapped

    def __call__(self, node):
        # err (and warnings) could be logged, which might be better..
        # This should still return None on invalid mapping.
        if node.keywords:
            if not self.allow_keywords:
                raise ValueError(f"{self.name} does not allow keywords.")
            else:
                mapped = self.validate_call(node.args, node.keywords)
                return self.replacement(mapped)
        else:
            mapped = self.validate_simple_call(node.args)
            return self.replacement(mapped)


def EnumerateBuilder(mapping):
    return ir.Zip((ir.AffineSeq(mapping["start"], None, ir.One), mapping["iterable"]))


def IterBuilder(mapping):
    return mapping["iterable"]


# def ReversedBuilder(arg):
#    if isinstance(arg, ir.AffineSeq):
#        if arg.stop is not None:
#            node = ir.AffineSeq(arg.stop, arg.start, negate_condition(arg.step))
#        else:
#            # must be enumerate
#            return "Cannot reverse enumerate type"
#    else:
#        node = ir.Reversed(arg)
#    return node


def ZipBuilder(node: ir.Call):
    """
    Zip takes an arbitrary number of positional arguments, something which we don't generally support.

    """
    assert (node.func.name == "zip")
    if node.keywords:
        raise ValueError("Zip does not accept keyword arguments.")
    args = tuple(arg for arg in node.args)
    return ir.Zip(args)


def RangeBuilder(node: ir.Call):
    """
    Range is a special case and handled separately from other call signatures,
    since it has to accommodate two distinct orderings without the use of keywords.

    These are:
        range(stop)
        range(start, stop, step: optional)

    Unlike the CPython version, this is only supported as part of a for loop.

    """

    argct = len(node.args)
    if not (0 < argct <= 3):
        return None, "Range call with incorrect number of arguments"
    elif node.keywords:
        return None, "Range does not support keyword arguments"
    if argct == 1:
        repl = ir.AffineSeq(ir.Zero, node.args[0], ir.One)
    elif argct == 2:
        repl = ir.AffineSeq(node.args[0], node.args[1], ir.One)
    else:
        repl = ir.AffineSeq(node.args[0], node.args[1], node.args[2])
    return repl


builders = {"enumerate": CallSpecialize(name="enumerate",
                                        args=("iterable", "start"),
                                        repl=EnumerateBuilder,
                                        defaults={"start": ir.Zero},
                                        allows_keywords=True),

            #  "reversed": CallSpecialize(name="reversed",
            #                           args=("object",),
            #                            repl=ReversedBuilder,
            #                            defaults={},
            #                            allows_keywords=False),

            "iter": CallSpecialize(name="iter",
                                   args=("iterable",),
                                   repl=IterBuilder,
                                   defaults={},
                                   allows_keywords=False),

            # Todo: orphaned
            # "len": CallSpecialize(name="len",
            #                      args=("obj",),
            #                      repl=LenBuilder,
            #                      defaults={},
            #                      allows_keywords=False),

            "range": RangeBuilder,

            "zip": ZipBuilder,
            }


def replace_builtin_call(node: ir.Call):
    handler = builders.get(node.func.name)
    if handler is None:
        return node
    return handler(node)
