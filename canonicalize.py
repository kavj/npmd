import numpy
import operator

from functools import singledispatchmethod, cached_property
from itertools import zip_longest

import ir
import type_resolution as tr
from errors import CompilerError
from utils import extract_name, wrap_input
from visitor import StmtTransformer


def type_from_numpy_type(t: type):
    if t == numpy.int32:
        return tr.Int32
    elif t == numpy.int64:
        return tr.Int64
    elif t == numpy.bool_:
        return tr.BoolType
    elif t == bool:
        return tr.BoolType
    elif t == numpy.float32:
        return tr.Float32
    elif t == numpy.float64:
        return tr.Float64
    elif t in (tr.Int32, tr.Int64, tr.Float32, tr.Float64):
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


class CallSpecialize:
    """
    Class to set up specialized call replacements. This is meant to be as simple as possible.
    Alternatively, it's possible to write a visitor that performs more involved replacements.

    This is forced to consider default arguments, because some builtins require it.
    It does not support * or ** signatures.

    Builders may raise exceptions on unsupported arguments, but they should explicitly
    recognize all named argument fields.

    Note: We may need to recognize optional fields, where None isn't immediately replaced.
          For fields that default to None, where None is immediately replaced, just use
          its replacement as a default instead.

    """

    def __init__(self, args, builder, allow_keywords):
        if len({*args}) < len(args):
            # relying on ordered dictionaries here is too error prone.
            msg = f"Call lowering has duplicate argument fields, {args} received."
            raise ValueError(msg)
        self.args = args
        self.allow_keywords = allow_keywords
        self.builder = builder

    @cached_property
    def max_arg_count(self):
        return len(self.args)

    @cached_property
    def default_count(self):
        return sum(1 for arg, default in self.args if default is not None)

    @cached_property
    def min_arg_count(self):
        return len(self.args) - self.default_count

    def __call__(self, node: ir.Call):
        name = node.func
        args = node.args
        keywords = node.keywords
        arg_count = len(args)
        kw_count = len(keywords)
        if not self.allow_keywords and kw_count > 0:
            kwargs = ", ".join(kw for (kw, _) in keywords)
            raise CompilerError(f"Function {name} does not allow keyword arguments. Received keyword arguments:"
                                f"{kwargs}.")
        mapped = {}
        unrecognized = set()
        duplicates = set()
        if arg_count + kw_count > self.max_arg_count:
            raise CompilerError(f"Function {name} has {self.max_arg_count} fields. "
                                f"{arg_count + kw_count} arguments were provided.")
        for kw, value in keywords:
            if kw in self.args:
                if kw in mapped:
                    duplicates.add(kw)
                else:
                    mapped[kw] = value
            else:
                unrecognized.add(kw)
        for (field, default), value in zip_longest(self.args.items(), args):
            if field in mapped:
                if value is not None:
                    duplicates.add(field)
                continue
            # builder should handle cases that may not be None
            mapped[field] = value
        if unrecognized:
            bad_kws = ", ".join(u for u in unrecognized)
            msg = f"Unrecognized fields: {bad_kws} in call to {name}."
            raise CompilerError(msg)
        elif duplicates:
            dupes = ", ".join(d for d in duplicates)
            msg = f"Duplicate fields: {dupes} in call to {name}."
            raise CompilerError(msg)
        repl = self.builder(mapped)
        return repl


def replace_array_init(kwargs):
    """
    Replace a call to zeros, ones, or empty with a specialized node.

    Only C ordering is supported at this time.

    "like" is unsupported, because it requires handling alternative implementations
    of the __array__ protocol.
    """
    order = extract_name(kwargs["order"])
    like = extract_name(kwargs["like"])
    if order != "C":
        msg = "Only C ordering is supported at this time."
        raise CompilerError(msg)
    elif like is not None:
        msg = "Numpy array initialization with 'like' parameter set is not yet supported."
        raise CompilerError(msg)
    dtype = type_from_numpy_type(kwargs["dtype"])
    shape = tuple(wrap_input(s) for s in kwargs["shape"])
    fill_value = ir.One
    return ir.ArrayInitSpec(shape, dtype, fill_value)


def replace_ones(kwargs):
    kwargs["fill_value"] = ir.One
    return replace_array_init(kwargs)


def replace_zeros(kwargs):
    kwargs["fill_value"] = ir.Zero
    return replace_array_init(kwargs)


def replace_empty(kwargs):
    kwargs["fill_value"] = None
    return replace_array_init(kwargs)


def replace_iter(kwargs):
    iterable = kwargs.get["iterable"]
    if iterable is None:
        msg = "iter is only supported with exactly one argument. No value found for iterable."
        raise CompilerError(msg)
    return iterable


def replace_len(node: ir.Call):
    if node.keywords:
        msg = "'len' does not allow keyword arguments."
        raise CompilerError(msg)
    nargs = len(node.args)
    if nargs != 1:
        msg = f"'len' accepts a single argument. {nargs} provided"
        raise CompilerError(msg)
    iterable, = node.args
    return ir.Length(iterable)


def replace_range(node: ir.Call):
    if node.keywords:
        msg = "Call to 'range' does not support keyword arguments."
        raise CompilerError(msg)
    nargs = len(node.args)
    # defaults
    start = ir.Zero
    step = ir.One
    if nargs == 1:
        stop, = node.args
    elif nargs == 2:
        start, stop = node.args
    elif nargs == 3:
        start, stop, step = node.args
    else:
        msg = f"Range accepts 1 to 3 arguments, {nargs} provided."
        raise CompilerError(msg)
    return ir.AffineSeq(start, stop, step)


def replace_enumerate(node: ir.Call):
    nargs = len(node.args) + len(node.keywords)
    if not (1 <= nargs <= 2):
        msg = f"'enumerate' accepts either 1 or 2 arguments, {nargs} provided."
        raise CompilerError(msg)
    if node.keywords:
        if node.args:
            key, value = node.keywords
            if key != "start":
                if key == "iterable":
                    msg = "keyword 'iterable' in call to enumerate shadows a positional argument."
                else:
                    msg = f"Unrecognized keyword {key} in call to enumerate"
                raise CompilerError(msg)
            start = value
            iterable, = node.args
        else:
            if len(node.keywords) == 1:
                key, iterable = node.keywords
                if key != "iterable":
                    msg = "Missing 'iterable' argument in call to enumerate"
                    raise CompilerError(msg)
                start = ir.Zero
            else:
                # dictionaries are not hashable, thus we have a tuple with explicit pairings
                # we should factor out a duplicate key check earlier on
                kwargs = node.keywords
                iterable = None
                start = None
                for key, value in kwargs:
                    if key == "iterable":
                        iterable = value
                    elif key == "start":
                        start = value
                    else:
                        msg = f"Unrecognized keyword {key} in call to enumerate."
                        raise CompilerError(msg)
                if iterable is None or start is None:
                    msg = f"bad keyword combination {kwargs[0][0]} {kwargs[0][1]} in call to enumerate, expected" \
                          f"'iterable' and 'start'"
                    raise CompilerError(msg)
        return ir.Enumerate(iterable, start)


def replace_zip(node: ir.Call):
    if node.keywords:
        msg = "zip does not support keyword arguments."
        raise CompilerError(msg)
    return ir.Zip(node.args)


replacers = {"enumerate": replace_enumerate,
             "iter": replace_iter,
             "len": replace_len,
             "range": replace_range,
             "zip": replace_zip,
             "numpy.empty": CallSpecialize({"shape": None, "dtype": numpy.float64, "order": "C", "like": None},
                                           replace_empty,
                                           allow_keywords=True),
             "numpy.ones": CallSpecialize({"shape": None, "dtype": numpy.float64, "order": "C", "like": None},
                                          replace_ones,
                                          allow_keywords=True),
             "numpy.zeros": CallSpecialize({"shape": None, "dtype": numpy.float64, "order": "C", "like": None},
                                           replace_zeros,
                                           allow_keywords=True)
             }


def replace_call(node: ir.Call):
    """
    AST level replacement for functions that are specialized internally.
    """
    func_name = extract_name(node)
    replacer = replacers.get(func_name)
    if replacer is None:
        if func_name.startswith("numpy."):
            msg = f"Invalid or unsupported numpy call: {func_name}."
            raise CompilerError(msg)
        return node
        # Todo: expand this for (any,all later
    # Todo: Some of the builtin calls follow strange overloading conventions
    #       so they had to be separated. I may rework this further, as I hate type checks.
    repl = replacer(node)
    return repl
