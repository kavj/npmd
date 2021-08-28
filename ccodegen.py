import textwrap

from contextlib import contextmanager
from functools import singledispatchmethod

import ir
from errors import CompilerError
from pretty_printing import pretty_formatter
from type_inference import ExprTypeInfer
from visitor import StmtVisitor

"""

most of the transforms are done elsewhere
This should load the appropriate ISA for simd ops and

This still needs a customized visitor to generate statements.
At this point, statements are serialized to the level expected by C code.

"""


class ctx:

    def __init__(self, uniform, decl_map, type_mapping, lltype_mapping):
        self.uniform_vars = uniform           # call invariant
        self.decl_map = decl_map              # indicates where variable names must be declared
        self.unique_call_names = set()        # qualified call names
        self.type_mapping = type_mapping      # type -> {vars of this type}
        self.lltype_mapping = lltype_mapping  # type -> {back end type}

    def get_req_headers(self):
        pass

    def get_raw_array_pointer_name(self, basetype):
        # move inside a class
        t = basetype.dtype if isinstance(basetype, (ir.ArrayRef, ir.ViewRef)) else basetype
        return self.lltype_mapping.get(t)

    def get_raw_scalar_name(self, basetype):
        return self.lltype_mapping.get(basetype)


def extract_leading_dim(array):
    return array.dims[0]


def enter_for_loop(header: ir.ForLoop, lltypes):
    loop_index = header.target
    counter = header.iterable
    # we'll need to add the exception code for illegal (here) step
    if counter.reversed:
        end_expr = ">"
        if counter.step == ir.One:
            step_expr = f"--{loop_index}"
        else:
            step_expr = f"{loop_index} -= {counter.step}"
    else:
        end_expr = "<"
        if counter.step == ir.One:
            step_expr = f"++{loop_index}"
        else:
            step_expr = f"{loop_index} += {counter.step}"

    return f"for({str(lltypes[loop_index])} {loop_index.name} = {counter.start}; {loop_index} {end_expr}" \
           f" {counter.stop}; {step_expr})"


def generate_expression(expr):
    return


def enter_while_loop(header: ir.WhileLoop):
    return f"while({generate_expression(header.test)})"


class code_generator:
    encoding: None
    indent: str
    decls: set
    type_map: dict

    def print(self):
        pass


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


class CCodeGen(StmtVisitor):

    # This is meant to be controlled by a codegen driver,
    # which manages opening/closing of a real or virtual destination file.

    def __init__(self, ctx):
        self.ctx = ctx
        self.infer_expr_type = ExprTypeInfer(self.ctx.types)
        self.format = pretty_formatter()
        self.printer = CodeEmitter()

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
    def scoping(self):
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
        stmt = f"for({target} = {start}; {target} < {stop}; {step_expr}"
        self.print_line(stmt)

        with self.scoping():
            self.visit(node.body)

