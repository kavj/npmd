import typing

from lib.errors import CompilerError
from lib.blocks import BasicBlock, FunctionContext
from lib.graph_walkers import find_branch_exit, find_loop_exit, get_reduced_graph, walk_graph
from lib.formatting import binop_ops, PrettyFormatter

from functools import singledispatchmethod
from contextlib import contextmanager

import lib.ir as ir


# Todo: Given the boolean refactoring, not should probably derive from BoolOp, similar to TRUTH.

# Todo: Pretty printer should provide most of the infrastructure for C code gen. For plain C, most of the statement
#       structure used is the same, so this should be handled as much as posible via different expression visitors.
#       I'll also need to check for differences in operator precedence.

# Note, python docs don't specify truth precedence, but it should match logical "not"


def format_error(msg: str, pos: ir.Position,
                 named: typing.Optional[typing.Dict] = None,
                 exprs: typing.Optional[typing.Iterable[ir.ValueRef]] = None):
    pf = PrettyFormatter()

    formatted_names = {pf(k): pf(v) for (k, v) in named.items()} if named is not None else ()
    formatted_exprs = {pf(e) for e in exprs} if exprs is not None else ()
    combined = "\n".join((str(pos), msg, str(formatted_names), str(formatted_exprs)))
    return combined


def format_docstring(ds: str, indent='    '):
    split_dst = [f'{indent}{line}' for line in ds.split('\n')]
    joined = '\n'.join(split_dst)
    reconstructed = f'{indent}"""\n{joined}\n{indent}"""'
    return reconstructed


class PrettyPrinter:
    """
    Pretty prints tree. 
    Inserts pass on empty if statements or for/while loops.

    """

    def __init__(self, single_indent="    ", print_annotations=True, allow_missing_type=True):
        self.indent = ""
        self._increment = len(single_indent)
        self._single_indent = single_indent
        self.print_annotations = print_annotations
        self.format = PrettyFormatter()
        self.allow_missing_type = allow_missing_type
        self.func = None
        self.graph = None
        self.exit_labels = None
        self.visited = None
        self.by_label = None

    @contextmanager
    def function_scope(self, func: FunctionContext):
        assert self.graph is None
        self.func = func
        self.graph = get_reduced_graph(func)
        self.visited = set()
        self.exit_labels = set()
        self.by_label = {block.label: block for block in walk_graph(self.graph)}
        yield
        self.func = None
        self.graph = None
        self.exit_labels = None
        self.visited = None
        self.by_label = None

    def __call__(self, func_ctx: FunctionContext):
        assert self.indent == ""
        with self.function_scope(func_ctx):
            self.visit(func_ctx.entry_point)

    def iter_successors(self, block: BasicBlock):
        if not block.terminated:
            for s in self.graph.successors(block):
                if s.label not in self.exit_labels and s not in self.visited:
                    yield s

    @contextmanager
    def indented(self):
        self.indent = f"{self.indent}{self._single_indent}"
        yield
        self.indent = self.indent[:-self._increment]

    def print_line(self, as_str):
        line = f"{self.indent}{as_str}"
        print(line)

    @singledispatchmethod
    def visit(self, node):
        msg = f"No method to pretty print node {node}."
        raise NotImplementedError(msg)

    @visit.register
    def _(self, node: BasicBlock):
        if node in self.visited:
            return
        self.visited.add(node)
        if node.is_function_entry:
            self.visit(node.first)
            body, = self.iter_successors(node)
            with self.indented():
                self.visit(body)
        elif node.is_loop_block:
            loop_header = node.first
            self.visit(loop_header)
            exit_block = find_loop_exit(self.func, node)
            self.exit_labels.add(exit_block)
            with self.indented():
                for possible_body in self.iter_successors(node):
                    if possible_body.depth == node.depth + 1:
                        self.visit(possible_body)
                        break
            if exit_block is not None:
                # exit_block = self.by_label[loop_header.exit_block]
                self.visit(exit_block)
        elif node.is_branch_block:
            branch_header = node.first
            if_branch = self.by_label[branch_header.if_branch]
            else_branch = self.by_label[branch_header.else_branch]
            test = self.format(branch_header.test)
            exit_block = find_branch_exit(self.func, node)
            if exit_block is not None:
                self.exit_labels.add(exit_block)
            self.print_line(f'if {test}:')
            with self.indented():
                # if we have a blank block at the beginning but it doesn't terminate the branch,
                # then continue visiting. still not great..
                if if_branch:
                    self.visit(if_branch)
                else:
                    self.print_line('pass')
            if else_branch:
                self.print_line('else:')
                with self.indented():
                    self.visit(else_branch)
            if exit_block is not None:
                self.visit(exit_block)
        else:
            for stmt in node.statements:
                self.visit(stmt)
            for s in self.iter_successors(node):
                self.visit(s)

    @visit.register
    def _(self, node: ir.ArrayFill):
        line = f'{self.format(node.array)}[...] = {self.format(node.fill_value)}'
        self.print_line(line)

    @visit.register
    def _(self, node: ir.Return):
        if isinstance(node.value, ir.NoneRef):
            self.print_line("return")
        else:
            expr = self.format(node.value)
            stmt = f"return {expr}"
            self.print_line(stmt)

    @visit.register
    def _(self, node: ir.Module):
        for f in node.functions:
            print('\n')
            self.visit(f)
            print('\n')

    @visit.register
    def _(self, node: ir.Function):
        name = node.name
        args = ", ".join(self.format(arg) for arg in node.args)
        header = f"def {name}({args}):"
        self.print_line(header)
        if node.docstring is not None:
            docstring = format_docstring(node.docstring)
            # docstring formatting is annoying here. We have to indent each line in the string,
            # which requires that we split
            self.print_line(docstring)

    @visit.register
    def _(self, node: list):
        for stmt in node:
            self.visit(stmt)

    @visit.register
    def _(self, node: ir.ForLoop):
        target = self.format(node.target)
        iterable = self.format(node.iterable)
        stmt = f"for {target} in {iterable}:"
        self.print_line(stmt)

    @visit.register
    def _(self, node: ir.WhileLoop):
        test = self.format(node.test)
        stmt = f"while {test}:"
        self.print_line(stmt)

    @visit.register
    def _(self, node: ir.IfElse):
        test = self.format(node.test)
        stmt = f"if {test}:"
        self.print_line(stmt)

    @visit.register
    def _(self, node: ir.Assign):
        target = node.target
        formatted_target = self.format(node.target)
        formatted_value = self.format(node.value)
        if self.print_annotations and isinstance(target, ir.NameRef):
            # type_ = self.symbols.check_type(target, allow_none=self.allow_missing_type)
            type_ = None
            if type_ is not None:
                # This is None if no typed symbol is registered
                # This will be an error later.
                formatted_target = f"{formatted_target}: {type_}"
        stmt = f"{formatted_target} = {formatted_value}"
        self.print_line(stmt)

    @visit.register
    def _(self, node: ir.InPlaceOp):
        left = node.target
        right = None
        for right in node.value.subexprs:
            # if there's a distinct one, pick it
            if right is not left:
                break
        assert right is not None
        op = binop_ops[type(node.value)]
        left = self.format(left)
        right = self.format(right)
        formatted = f'{left} {op}= {right}'
        # Todo: needs update
        self.print_line(formatted)

    @visit.register
    def _(self, node: ir.Continue):
        self.print_line("continue")

    @visit.register
    def _(self, node: ir.Break):
        self.print_line("break")

    @visit.register
    def _(self, node: ir.SingleExpr):
        expr = self.format(node.value)
        self.print_line(expr)


class DebugPrinter:

    def __init__(self, max_len: int = 40):
        self.max_len = max_len
        self.formatter = PrettyFormatter()

    def format(self, node):
        return self.formatter.visit(node)

    def visit(self, node: typing.Union[ir.StmtBase, ir.Function], indent='    '):
        if not isinstance(node, (ir.StmtBase, ir.Function)):
            msg = f'Debug printer expects a statement or function. Received: "{node}"'
            raise CompilerError(msg)
        if isinstance(node, ir.Function):
            # Todo: Functions could get a position attributes..
            args = [arg.name for arg in node.args]
            arg_str = ", ".join(args)
            formatted = f'{node.name}({arg_str})'
        else:
            formatted = self.format(node)
        if node.docstring is not None:
            docstring = f'"""\n{node.docstring}\n"""'
            formatted = f'{formatted}\n{docstring}'

        if len(formatted) > self.max_len:
            formatted = f'{formatted[:self.max_len]}...'
        # Todo: adding this to func later..
        pos = node.pos.line_begin if isinstance(node, ir.StmtBase) else ''
        formatted = f'{indent}line: {pos}, {formatted}'
        print(formatted)

    def print_block(self, block: typing.Iterable[ir.StmtBase]):
        for stmt in block:
            self.visit(stmt)
