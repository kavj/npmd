import itertools
import operator
import typing
from functools import singledispatch, singledispatchmethod

import ir
from visitor import TransformBase, walk_branches


def negate_condition(node):
    repl = ir.UnaryOp(node, "not")
    if node.constant:
        repl = ir.BoolNode(operator.invert(operator.truth(node)))
    elif isinstance(node, ir.UnaryOp):
        if node.op == "not":
            repl = node.operand
    elif isinstance(node, ir.BinOp):
        # Only fold cases with a single operator that has a negated form.
        # Otherwise we have to worry about edge cases involving unordered operands.
        if node.op == "==":
            repl = ir.BinOp(node.left, node.right, "!=")
        elif node.op == "!=":
            repl = ir.BinOp(node.left, node.right, "==")
    else:
        repl = ir.UnaryOp(node, "not")
    return repl


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
    if isinstance(node.test, ir.ValueRef):
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
def _(node: ir.Continue):
    return True


@terminates_control_flow.register
def _(node: ir.Return):
    return True


@terminates_control_flow.register
def _(node: ir.Break):
    return True


@terminates_control_flow.register
def _(node: ir.StmtBase):
    return False


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
        return ir.ForLoop(node.target, node.iterable, repl, node.pos)

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
        target = node.target
        iterable = node.iterable
        body = self.visit(node.body)
        pos = node.pos
        header = self.loops.pop()
        assert (header is node)
        return ir.ForLoop(target, iterable, body, pos)

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
            raise ValueError(f"Function {self.name} does not allow keyword arguments.")
        mapped = {}
        unrecognized = set()
        duplicates = set()
        missing = set()
        if arg_count + kw_count > self.max_arg_count:
            raise ValueError(f"Function {self.name} has {self.max_arg_count} fields. "
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
            raise ValueError(f"unrecognized field {u} in call to {self.name}")
        return self.replacement(mapped)

    def validate_simple_call(self, args):
        # Check call with no keywords
        arg_count = len(args)
        mapped = {}
        if arg_count > self.max_arg_count:
            raise ValueError(f"Signature for {self.name} accepts {self.max_arg_count} arguments. "
                               f"{arg_count} arguments received.")
        elif arg_count < self.min_arg_count:
            raise ValueError(f"Signature for {self.name} expects at least {self.min_arg_count} arguments, "
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
    return ir.Zip((ir.Counter(mapping["start"], None, ir.IntNode(1)), mapping["iterable"]))


def IterBuilder(mapping):
    return mapping["iterable"]


def ReversedBuilder(arg):
    if isinstance(arg, ir.Counter):
        if arg.stop is not None:
            node = ir.Counter(arg.stop, arg.start, negate_condition(arg.step))
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
    assert (node.funcname == "zip")
    if node.keywords:
        raise ValueError("Zip does not accept keyword arguments.")

    return ir.Zip(tuple(arg for arg in node.args))


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
        repl = ir.Counter(ir.IntNode(0), node.args[0], ir.IntNode(1))
    elif argct == 2:
        repl = ir.Counter(node.args[0], node.args[1], ir.IntNode(1))
    else:
        repl = ir.Counter(node.args[0], node.args[1], node.args[2])
    return repl


builders = {"enumerate": CallSpecialize(name="enumerate",
                                        args=("iterable", "start"),
                                        repl=EnumerateBuilder,
                                        defaults={"start": ir.IntNode(0)},
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
    handler = builders.get(node.funcname)
    if handler is None:
        return node
    return handler(node)
