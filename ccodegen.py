import textwrap

from contextlib import contextmanager
from functools import singledispatchmethod

import ir
from errors import CompilerError
from pretty_printing import pretty_formatter, binop_ordering
from type_inference import ExprTypeInfer
from visitor import StmtVisitor, ExpressionVisitor

"""

most of the transforms are done elsewhere
This should load the appropriate ISA for simd ops and

This still needs a customized visitor to generate statements.
At this point, statements are serialized to the level expected by C code.

"""


class CTypeResolver:

    # Take type from context, map to C data type.
    # must handle simd types

    def __init__(self, ctx):
        self.ctx = ctx


class CodeEmitter:

    indent_type = "    "

    def __init__(self, context, file, indent="    ", max_line_width=70):
        self.ctx = context
        self.file = file
        self.indent = ""
        self.single_indent = indent
        self.max_line_width = max_line_width
        self.tree = None
        self.dest = None

    def __call__(self, tree, dest):
        # Todo: work these into context manager
        self.tree = tree
        self.dest = dest

    @contextmanager
    def indented(self):
        indent_len = len(self.indent)
        char_count = len(self.single_indent)
        self.indent = f"{self.indent}{self.single_indent}"
        yield
        self.indent = self.indent[:-char_count]
        if indent_len != len(self.indent):
            raise RuntimeError

    def print_line(self, line):
        line = textwrap.wrap(line, width=self.max_line_width)
        print(line, file=self.dest)


class ExpressionResolver(ExpressionVisitor):
    # resolve expression, given operands
    # this is needed as simd types don't reliably
    # work with arithmetic operators, across all compilers.

    # This also needs to deal with adding parentheses
    # maybe refactor from pretty_printing or import from it.

    @singledispatchmethod
    def visit(self, expr):
        raise NotImplementedError


def else_is_elif(stmt: ir.IfElse):
    if len(stmt.else_branch) == 1:
        if isinstance(stmt.else_branch[0], ir.IfElse):
            return True
    return False


class CCodeGen(StmtVisitor):

    # This is meant to be controlled by a codegen driver,
    # which manages opening/closing of a real or virtual destination file.

    def __init__(self, ctx, dest):
        self.ctx = ctx
        self.infer_expr_type = ExprTypeInfer(self.ctx.types)
        self.format = pretty_formatter()
        self.printer = CodeEmitter(ctx, dest)

    @contextmanager
    def function_context(self):
        # This should load function specific types from module
        # context and set up variables
        yield

    def __call__(self, func: ir.Function):
        self.declared = set()
        self.visit(func)

    def retrieve_type(self, ref):
        return self.ctx.retrieve_type(ref)

    def print_line(self, stmt):
        self.printer.print_line(stmt)

    @contextmanager
    def indented(self):
        with self.printer.indented():
            yield

    @singledispatchmethod
    def visit(self, node):
        raise NotImplementedError

    @visit.register
    def _(self, node: ir.Assign):
        # check types
        rhs_type = self.retrieve_type(node.value)
        lhs_type = self.retrieve_type(node.target)
        if lhs_type != rhs_type:
            raise CompilerError

        target = self.format(node.target)
        value = self.format(node.value)
        if node.target in self.declared:
            stmt = f"{target} = {value};"
        else:
            # Todo: declaration needs to obtain a
            #     corresponding destination language type, in this case C.
            raise NotImplementedError
        self.print_line(stmt)

    def visit_elif(self, node: ir.IfElse):
        test = self.visit(node.test)
        branch_head = f"else if({test}){'{'}"
        self.print_line(branch_head)
        with self.indented():
            self.visit(node.if_branch)
        self.print_line("}")
        if else_is_elif(node):
            self.visit_elif(node.else_branch[0])
        elif node.else_branch:
            self.print_line("else{")
            with self.indented():
                self.visit(node.else_branch)
            self.print_line("}")

    @visit.register
    def _(self, node: ir.IfElse):
        test = self.visit(node.test)
        branch_head = f"if({test}){'{'}"
        self.print_line(branch_head)
        with self.indented():
            self.visit(node.if_branch)
        self.print_line("}")
        if else_is_elif(node):
            self.visit_elif(node)
        elif node.else_branch:
            self.print_line("else{")
            with self.indented():
                self.visit(node.else_branch)
            self.print_line("}")

    @visit.register
    def _(self, node: ir.ForLoop):
        # Printing is more restrictive, check that
        # we have a supported loop structure
        assert isinstance(node.target, ir.NameRef)
        assert isinstance(node.iterable, (ir.AffineSeq, ir.Reversed))
        # check for unit step
        if node.target in self.declared:
            pass
        else:
            pass
        # check whether we can use ++ op
        # insufficient support for reversed() thus far.
        target = self.format(node.target)
        if node.iterable.step == ir.One:
            step_expr = f"++{target}"
        else:
            increm_by = self.format(node.iterable.step)
            step_expr = f"{target} += {increm_by}"
        start = self.format(node.iterable.start)
        stop = self.format(node.iterable.stop)
        stmt = f"for({target} = {start}; {target} < {stop}; {step_expr}){'{'}"
        self.print_line(stmt)
        with self.indented():
            self.visit(node.body)
        self.print_line("}")
