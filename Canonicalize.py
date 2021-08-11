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


def remove_dead_branches(node: ir.IfElse) -> typing.Union[ir.IfElse, list]:
    if node.test.constant:
        if operator.truth(node.test):
            node = node.if_branch
        else:
            node = node.else_branch
    return node


def contains_break(entry):
    raise NotImplementedError


def terminates_control_flow(node):
    # Note: It's probably best to disallow return statements in
    # for and while loops, because if any alternative return does
    # not immediately follow the loop body, you get a very complicated
    # and potentially inefficient output. Because of that, I'm removing it here.
    if isinstance(node, list):
        return len(node) > 0 and terminates_control_flow(node[-1])
    elif isinstance(node, ir.StmtBase):
        if isinstance(node, (ir.Continue, ir.Break, ir.Return)):
            return True
        elif isinstance(node, ir.IfElse):
            return terminates_control_flow(node.if_branch) and terminates_control_flow(node.else_branch)
    else:
        msg = f"Internal Error: No method to check control flow termination for node of type {type(node)}."
        raise NotImplementedError(msg)


def find_unterminated_path(stmts):
    if not isinstance(stmts, list):
        raise TypeError("Internal Error: expected a list of statements")
    if len(stmts) > 0:
        last = stmts[-1]
        if isinstance(last, (ir.Continue, ir.Break, ir.Return)):
            return TerminatedPath()
        elif isinstance(last, ir.IfElse):
            if_branch = find_unterminated_path(last.if_branch)
            else_branch = find_unterminated_path(last.else_branch)
            if_is_terminated = isinstance(if_branch, TerminatedPath)
            else_is_terminated = isinstance(else_branch, TerminatedPath)
            if if_is_terminated and else_is_terminated:
                return TerminatedPath()
            elif if_is_terminated:
                return else_branch
            elif else_is_terminated:
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

    def __init__(self):
        self.innermost_loop = None
        self.body = None

    @property
    def within_loop(self):
        return self.innermost_loop is not None

    @contextmanager
    def enclosing_loop(self, header):
        stashed = self.enclosing_loop
        yield
        assert self.innermost_loop is header
        self.body = stashed

    def __call__(self, node):
        if self.within_loop:
            raise RuntimeError("Internal Error: Entering visitor from inconsistent state.")
        node = self.visit(node)
        if self.within_loop:
            raise RuntimeError("Internal Error: Exiting visitor from inconsistent state.")
        return node

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.ForLoop):
        with self.enclosing_loop(node):
            body = self.visit(node.body)
            if body != node.body:
                node = ir.ForLoop(node.target, node.iterable, body, node.pos)
        return node

    @visit.register
    def _(self, node: ir.WhileLoop):
        test = node.test
        if test.constant:
            if not operator.truth(test):
                return []
        with self.enclosing_loop(node):
            body = self.visit(node.body)
            if body != node.body:
                node = ir.WhileLoop(node.test, body, node.pos)
        return node

    @visit.register
    def _(self, node: ir.IfElse):
        node = remove_dead_branches(node)
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
                stmt = remove_dead_branches(stmt)
                if isinstance(stmt, ir.IfElse):
                    append_to.append(stmt)
                    unterminated_if = find_unterminated_path(stmt.if_branch)
                    unterminated_else = find_unterminated_path(stmt.else_branch)
                    if isinstance(unterminated_if, TerminatedPath):
                        if isinstance(unterminated_else, TerminatedPath):
                            # remaining statements are unreachable
                            break
                        else:
                            append_to = unterminated_else
                    elif isinstance(unterminated_else, TerminatedPath):
                        append_to = unterminated_if
                else:
                    # visit remaining live branch
                    branch_stmts = self.visit(stmt)
                    unterminated = find_unterminated_path(branch_stmts)
                    append_to.extend(branch_stmts)
                    if unterminated is not branch_stmts:
                        # update append_to if the remaining live branch here
                        # contains terminated sub-paths
                        append_to = unterminated
            else:
                stmt = self.visit(stmt)
                append_to.append(stmt)
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


def ReversedBuilder(arg):
    if isinstance(arg, ir.AffineSeq):
        if arg.stop is not None:
            node = ir.AffineSeq(arg.stop, arg.start, negate_condition(arg.step))
        else:
            # must be enumerate
            return "Cannot reverse enumerate type"
    else:
        node = ir.Reversed(arg)
    return node


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

            "reversed": CallSpecialize(name="reversed",
                                       args=("object",),
                                       repl=ReversedBuilder,
                                       defaults={},
                                       allows_keywords=False),

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
