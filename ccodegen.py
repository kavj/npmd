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


def map_internal_type_to_c99_type(type_):
    raise NotImplementedError


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


def format_header(prefix, cond):
    if cond is None:
        return f"{prefix}{'{'}"
    else:
        return f"{prefix} ({cond}){'{'}"


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
        self._declared = set()
        self.visit(func)

    def declared(self, ref: ir.NameRef):
        assert isinstance(ref, (ir.NameRef, ir.Subscript))
        return ref in self._declared

    def check_type(self, ref):
        return self.ctx.retrieve_type(ref)

    def print_line(self, stmt):
        self.printer.print_line(stmt)

    def format_lvalue_ref(self, expr):
        if isinstance(expr, ir.NameRef):
            formatted = self.format(expr)
            if not self.declared(expr):
                type_ = self.check_type(expr)
                # subject to change
                formatted = f"{type_} {formatted}"
            return formatted

    @contextmanager
    def scoped(self, prefix, cond):
        if cond is None:
            line = f"{prefix}{'{'}"
        else:
            line = f"{prefix} ({cond}){'{'}"
        self.print_line(line)
        with self.printer.indented():
            yield
        self.print_line("}")

    @singledispatchmethod
    def visit(self, node):
        raise NotImplementedError

    @visit.register
    def _(self, node: ir.Assign):
        # check types
        rhs_type = self.check_type(node.value)
        lhs_type = self.check_type(node.target)
        if lhs_type != rhs_type:
            raise CompilerError

        target = self.format(node.target)
        value = self.format(node.value)

        # Todo: need to determine how much of the numpy api should be directly exposed.
        #    At present, I am guessing anything that doesn't require the gil may be exposed,
        #    unless the use of restrict or __restrict__ becomes necessary
        #    (supported on all compilers that compile CPython).
        if isinstance(node.target, ir.NameRef) and node.target not in self.declared:
            # For now, assume C99 back end,
            # compliant with PEP 7
            type_ = self.ctx.get_type(target)
            ctype_ = map_internal_type_to_c99_type(type_)
            target = f"{ctype_} {target}"
        stmt = f"{target} = {value};"
        self.print_line(stmt)

    def visit_elif(self, node: ir.IfElse):
        test = self.visit(node.test)
        with self.scoped("else if", test):
            self.visit(node.if_branch)
        if else_is_elif(node):
            self.visit_elif(node.else_branch[0])
        elif node.else_branch:
            with self.scoped("else", None):
                self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.IfElse):
        test = self.visit(node.test)
        with self.scoped("if", test):
            self.visit(node.if_branch)
        if else_is_elif(node):
            self.visit_elif(node)
        elif node.else_branch:
            with self.scoped("else", None):
                self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.ForLoop):
        # Printing is more restrictive, check that
        # we have a supported loop structure
        assert isinstance(node.target, ir.NameRef)
        assert isinstance(node.iterable, (ir.AffineSeq, ir.Reversed))
        # check for unit step
        target = self.format_lvalue_ref(node.target)
        # check whether we can use ++ op
        # insufficient support for reversed() thus far.
        if node.iterable.step == ir.One:
            step_expr = f"++{target}"
        else:
            increm_by = self.format(node.iterable.step)
            step_expr = f"{target} += {increm_by}"
        start = self.format(node.iterable.start)
        stop = self.format(node.iterable.stop)
        # implements range semantics with forward step, with the caveat
        # that any escaping value of target must be copied out in the loop body
        cond = f"{target} = {start}; {target} < {stop}; {step_expr}"
        with self.scoped("for", cond):
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.WhileLoop):
        cond = self.format(node.test)
        with self.scoped("while", cond):
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.Break):
        self.print_line("break;")

    @visit.register
    def _(self, node: ir.Continue):
        self.print_line("continue;")

    @visit.register
    def _(self, node: ir.Assign):
        target = self.format(node.target)
        value = self.format(node.value)
        if node.in_place:
            assert isinstance(node.value, ir.BinOp)
            op = node.value.op
        else:
            op = "="
        line = f"{target} {op} {value};"
        self.print_line(line)

    @visit.register
    def _(self, node: ir.SingleExpr):
        expr = self.format(node.expr)
        line = f"{expr};"
        self.print_line(line)
