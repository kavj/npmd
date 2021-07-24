import ctypes
import numpy as np
import textwrap
import tempfile

from contextlib import ContextDecorator
from functools import singledispatchmethod

import ir
from visitor import StmtVisitor

"""

most of the transforms are done elsewhere
This should load the appropriate ISA for simd ops and

This still needs a customized visitor to generate statements.
At this point, statements are serialized to the level expected by C code.

"""


def build_lltypes():
    # bool is malleable, we may cast it implicitly
    lltypes = {bool: "bool"}

    # we don't have a ctypes type for int
    # It's generally 2s complement and 4 bytes. Any exceptions need
    # to be tracked according to the type returned by sys.system or similar
    for t, sz in ((np.int32, 4), (np.int64, 8)):
        if ctypes.sizeof(ctypes.c_long) == sz:
            lltypes[t] = "long"
        elif ctypes.sizeof(ctypes.c_longlong) == sz:
            lltypes[t] = "long long"
    for t, sz in ((np.float32, 4), (np.float64, 8)):
        if ctypes.sizeof(ctypes.c_float) == sz:
            lltypes[t] = "float"
        elif ctypes.sizeof(ctypes.c_double) == sz:
            lltypes[t] = "double"
    return lltypes


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


class mangler:
    """
    sets up function names by argument specialization
    """
    pass

    def mangle_name(self, func):
        return


class FuncWrapperGen:
    func: ir.Function
    mangler_: mangler

    @property
    def mangled(self):
        return self.mangler_.mangle_name(self.func.name)


def enter_func(func, lltypes, return_type):
    func_name = extract_name(func.name)
    rt = lltypes.get(return_type)
    assert rt is not None
    args = (f"{lltypes[arg]} {extract_name(arg)}" for arg in func.args)
    return f"{rt} {func_name} ({', '.join(arg for arg in args)})"


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


class indent_manager(ContextDecorator):
    # tracks indentation
    pass


class string_builder:
    # should maybe be a base class
    def __init__(self, types, encoding, linewrap=True, indent="    ", indent_level=0):
        self.encoding = encoding
        self.types = types
        self.linewrap = linewrap
        self.indent = indent
        self.indent_level = indent_level
        self.prefix = self.indent

    def build_declarations(self):
        # used to build declarations for all existing types at this point
        decls = []
        for t, names in self.types.items():
            decls.append(f"{str(t)} {', '.join(name for name in names)};\n")
        return decls

    def print(self, stmt):
        pass

    @singledispatchmethod
    def build_string(self, node, *args):
        raise NotImplementedError


class scope_entry(ContextDecorator):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class LoweringBuilder(StmtVisitor):

    def __call__(self, entry, ctx_):
        self.context = ctx_

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.ForLoop):
        pass
        # bounds = lower_for_loop_header_bounds(node, self.ctx.symbols)

    @visit.register
    def _(self, node: list):
        repl = []
        for stmt in node:
            if isinstance(stmt, ir.ForLoop):
                pass
                # bounds = lower_for_loop_header_bounds(node, self.ctx.symbols)
                loop_index = self.context.make_unique_name()
                # These are all affine statements, thus simple
                # entry_prologue = lower_iterator_access_funcs(stmt, loop_index)
                # body = self.visit(stmt.body)
                # header = ir.ForLoop([(loop_index, bounds)], body, stmt.pos)
                # repl.append(header)
            # elif isinstance()
