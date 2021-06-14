import ctypes
import numpy as np

from contextlib import ContextDecorator
from functools import singledispatchmethod

import ir
import loopanalysis
from lowering import extract_name
from visitor import VisitorBase, walk_all

"""

most of the transforms are done elsewhere
This should load the appropriate ISA for simd ops and

This still needs a customized visitor to generate statements.
At this point, statements are serialized to the level expected by C code.

"""


def get_req_headers(types):
    pass


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


class mangler:
    """
    sets up function names by argument specialization
    """
    pass


def build_symbols(entry):
    # grabs all names that can be declared at outmermost scope
    names = set()
    if isinstance(entry, ir.Function):
        for arg in entry.args:
            names.add(extract_name(arg))
    for stmt in walk_all(entry):
        if isinstance(stmt, ir.Assign):
            if isinstance(stmt.target, ir.NameRef):
                names.add(extract_name(stmt.target))
        elif isinstance(stmt, ir.ForLoop):
            for target, _ in stmt.walk_assignments():
                names.add(extract_name(target))
    return names


def requires_cast(expr, type_info):
    pass


def make_loop_exprs(header):
    """
    Makes simple bounds
    For each array, push either stride expression or subscript

    """
    pass


def mangle_name(func):
    return


class FuncWrapperGen:
    func: ir.Function

    @property
    def mangled(self):
        return mangle_name(self.func.name)


def get_raw_array_pointer_name(basetype):
    # move inside a class
    lltypes = build_lltypes()
    t = basetype.dtype if isinstance(basetype, (ir.ArrayRef, ir.ViewRef)) else basetype
    return lltypes.get(t)


def get_raw_scalar_name(basetype):
    lltypes = build_lltypes()
    return lltypes.get(basetype)


def make_func_header_string(func, types, return_type, array_dim_type):
    array_params = set()
    args = func.args
    for arg in args:
        t = types.get(arg)
        if isinstance(t, ir.ArrayRef):
            for value in t.dims:
                # append symbolic parameters
                if isinstance(value, (str, ir.NameRef)):
                    array_params.add(value)

    lltypes = build_lltypes()
    func_name = extract_name(func.name)
    rt = lltypes.get(return_type)
    assert rt is not None
    added = set()
    textual = []
    for arg in args:
        t = types.get(arg)
        arg = extract_name(arg)
        assert t is not None
        textual.append(f"{t} {arg}")
    for arg in array_params:
        arg = extract_name(arg)
        if arg in added:
            raise ValueError(f"Array input parameter {arg} shadows argument name")
        textual.append(f"{array_dim_type} {arg}")
    sig = f"{rt} {func_name}({','.join(arg for arg in textual)})"
    # Todo: make this wrap if necessary
    return sig


def extract_leading_dim(array):
    return array.dims[0]


def generate_for_loop_entry(header: ir.ForLoop, ctx):
    loop_index_name = ctx.symbols.generate_name()
    # this needs to actually generate a min over array params
    # for loop bounds

    array_refs = {it for target, it in header.walk_assignments() if not isinstance(it, ir.Counter)}

    # These take a trivial assignment from the loop index
    normalized_counters = loopanalysis.normalized_counters(header)
    # strided = get_strided_refs(header, types)

    # return header_text, body_assignments


def generate_expression(expr):
    return


def generate_while_loop_entry(header: ir.WhileLoop):
    return f"while({generate_expression(header.test)})"


# this should be context managed, like with predicated_scope()
# type observer
def predicated_scope():
    pass


class scalar_code_printer(VisitorBase):
    def __call__(self, entry, declared):
        # for simplicity, declare
        # at function entry unless single use
        self.declared = declared
        assert isinstance(entry, ir.Function)


class vector_code_printer(VisitorBase):
    # We need to split this into its own thing
    pass


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

    @build_string.register
    def _(self, node: ir.IfElse):
        return f"if({self.build_string(node.test)})"

    @build_string.register
    def _(self, node: ir.ForLoop):
        assign_index = -1
        for assign_index, (_,_) in node.walk_assignments():
            pass
        if assign_index == 0:
            raise ValueError
        loop_index, counter = next(node.walk_assignments())
        target_type = self.types[loop_index]
        self.print(f"for({target_type}{loop_index} = {counter.start}; {loop_index} < {counter.stop}; ++{loop_index})")

    @build_string.register
    def _(self, node: ir.WhileLoop):
        self.print(f"while({self.build_string(node.test)})")

    @build_string.register
    def _(self, node: ir.Assign, types, declare=False):
        target_type = types[node.target]
        expr_type = types[node.value]
        if node.in_place:
            assert not declare
            return f"{self.build_string(node.value)};"
        else:
            if declare:
                decl = f"{str(types[node.target])} "
            else:
                decl = ""
            if target_type == expr_type:
                # actually needs formatting but whatever
                stmt = f"{decl}{node.target} = {node.value}"
                pass
            else:
                pass

    @build_string.register
    def _(self, node: ir.IfExpr):
        pass

    @build_string.register
    def _(self, node: ir.BinOp):
        pass

    @build_string.register
    def _(self, node: ir.UnaryOp):
        pass


class scope_entry(ContextDecorator):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Builder(VisitorBase):

    def __call__(self, entry, printer, ctx):
        self.printer = printer
        self.ctx = ctx

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Function):
        header = make_func_header_string(node, self.ctx.types, self.ctx.return_type, self.ctx.array_dim_type)
        self.printer.print(header)
        with scope_entry():
            self.visit(node.body)

    @visit.register
    def _(self, node: list):
        for stmt in node:
            self.visit(stmt)

    @visit.register
    def _(self, node: ir.WhileLoop):
        # print while header
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.ForLoop):
        # Todo: rewrite, previous one was terrible
        raise NotImplementedError

    @visit.register
    def _(self, node: ir.IfElse):
        # print header
        with scope_entry:
            self.visit(node.if_branch)
        if node.else_branch:
            with scope_entry:
                self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.Assign):
        pass

    @visit.register
    def _(self, node: ir.BinOp):
        pass


class compile_driver(VisitorBase):
    pass
