import operator
from functools import singledispatchmethod

import ir
from errors import CompilerError
from visitor import StmtTransformer


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


def replace_builtin_call(node: ir.Call):
    name = node.func.name
    if name == "enumerate":
        args = node.args
        kws = node.keywords
        nargs = len(args)
        nkws = len(kws)
        if not (1 <= nargs + nkws <= 2):
            msg = f"In call to enumerate: {node}, enumerate requires exactly 1 or 2 arguments, {nargs + nkws} " \
                  f"arguments given."
            raise CompilerError(msg)
        elif nargs == 1:
            iterable, = args
            if kws:
                if len(kws) != 1:
                    msg = f"Call to enumerate: {node} contains too many keyword arguments."
                    raise CompilerError(msg)
                else:
                    (kw, start), = kws
                    if kw != "start":
                        msg = f"Call to enumerate: {node} has unexpected keyword argument '{kw}'"
                        raise CompilerError(msg)
            else:
                start = ir.Zero
        elif nargs == 2:
            if kws:
                kws = tuple(k for (k, _) in kws)
                msg = f"Call to enumerate with 2 positional arguments contains unexpected keyword arguments {kws}."
                raise CompilerError(msg)
            iterable, start = args
        else:
            raise CompilerError
        return ir.Enumerate(iterable, start)
    if name == "iter":
        if node.args:
            if len(node.args) != 1 or node.keywords:
                msg = "Only the single argument form of iter: 'iter(iterable)' is supported"
                raise CompilerError(msg)
        elif node.keywords:
            if len(node.keywords) > 1:
                msg = f"The only supported keyword argument form of iter that is supported is 'iter(iterable)', " \
                      f"received 2 or more keyword arguments."
                raise CompilerError(msg)
            kw, value = node.keywords
            if kw != "iterable":
                msg = "Unrecognized keyword {kw} in call to iter."
                raise CompilerError(msg)
        # no call specialization. within a loop this reduces to a noop, elsewhere it's unsupported
        return node
    elif name == "len":
        if node.keywords:
            msg = f"len does not support keyword arguments"
            raise CompilerError(msg)
        elif len(node.args) != 1:
            msg = f"len expects exactly one positional argument, {len(node.args)} arguments received."
            raise CompilerError(msg)
        arg, = node.args
        return ir.Length(arg)
    elif name == "range":
        if node.keywords:
            msg = "Range does not support keyword arguments."
            raise CompilerError(msg)
        nargs = len(node.args)
        if not (1 <= nargs <= 3):
            msg = f"Range expects 1 to 3 arguments, {len(node.args)} given."
            raise CompilerError(msg)
        if nargs == 1:
            start = ir.Zero
            stop, = node.args
            step = ir.One
        elif nargs == 2:
            start, stop = node.args
            step = ir.One
        else:
            start, stop, step = node.args
        return ir.AffineSeq(start, stop, step)
    elif name == "zip":
        if node.keywords:
            msg = "Zip does not support keywords."
            raise CompilerError(msg)
        return ir.Zip(node.args)
    else:
        return node
