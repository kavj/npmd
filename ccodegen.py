import ctypes
import numpy as np
import operator

from collections import deque

import ir
import loopanalysis
from lowering import extract_name, get_strided_refs
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


def generate_for_loop_entry(header: ir.ForLoop, types, symbols, array_dim_type):
    # needs to be able to generate loop index names
    # will need to later add something for liveness info
    header_text = []
    body_stmts = []
    loop_index_name = symbols.generate_name()
    # this needs to actually generate a min over array params
    # for loop bounds
    array_refs = {it for target, it in header.walk_assignments() if not isinstance(it, ir.Counter)}
    # These take a trivial assignment from the loop index
    normalized_counters = loopanalysis.normalized_counters(header)
    strided = get_strided_refs(header, types)

    # return header_text, body_assignments


def generate_expression(expr):
    return


def generate_while_loop_entry(header: ir.WhileLoop):
    return f"while({generate_expression(header.test)})"


def flatten_branches(header: ir.IfElse):
    """
    return something that matches an elif structure
    the notion is needed to distinguish these from independently taken branches

    """
    if (operator.truth(header.else_branch)
            and len(header.else_branch) == 1
            and isinstance(header.else_branch[0], ir.IfElse)):
        if_stmts = ir.IfElse(header.test, header.if_branch, [], header.pos)
        elif_ = flatten_branches(header.else_branch[0])
        elif_[0].is_elif = True
        elif_.appendleft(if_stmts)
    else:
        if_stmts = deque()
        if_stmts.append(header)
    return if_stmts


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


class compile_driver(VisitorBase):
    pass
