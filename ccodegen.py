import itertools
import numpy as np
import pathlib
import textwrap

from collections import defaultdict
from contextlib import contextmanager
from functools import singledispatchmethod
from typing import Dict, Iterable

import ir

from errors import CompilerError
from pretty_printing import binop_ordering, pretty_formatter
from symbol_table import symbol_table
from type_resolution import ExprTypeInfer
from utils import extract_name
from visitor import ExpressionVisitor, StmtVisitor

"""

most of the transforms are done elsewhere
This should load the appropriate ISA for simd ops and

This still needs a customized visitor to generate statements.
At this point, statements are serialized to the level expected by C code.

"""

std_include_headers = ("Python.h", "numpy/arrayobject.h", "stdint.h", "stdbool.h", "math.h")

# initial function name mangling abi
# scalar types use a single letter + bit count
# This is meant to disambiguate multiple lowerings based on arguments.

scalar_type_mangling = {
    ir.Int32: "i32",
    ir.Int64: "i64",
    ir.Float32: "f32",
    ir.Float64: "f64",
    ir.Predicate32: "p32",
    ir.Predicate64: "p64",
    ir.BoolType: "b8"
}

scalar_type_map = {ir.Int32: "int32_t",
                   ir.Int64: "int64_t",
                   ir.Float32: "float",
                   ir.Float64: "double",
                   ir.Predicate32: "bool",
                   ir.Predicate64: "bool",
                   ir.BoolType: "bool"}

npy_scalar_type_map = {ir.Int32: "npy_int32",
                       ir.Int64: "npy_int64",
                       ir.Float32: "float",
                       ir.Float64: "double",
                       ir.BoolType: "bool"}

npy_c_type_codes = {
    np.dtype("bool"): "NPY_BOOL",
    np.dtype("int32"): "NPY_INT32",
    np.dtype("int64"): "NPY_INT64",
    np.dtype("float32"): "NPY_FLOAT32",
    np.dtype("float64"): "NPY_FLOAT64",
}


npy_scalar_unwrap_funcs = {
    ir.Int32: "unwrap_int",
    ir.Int64: "unwrap_longlong",
    ir.Float32: "unwrap_float",
    ir.Float64: "unwrap_double",
    ir.BoolType: "unwrap_bool"
}

# These are used for array initialization
npy_dtype_codes = {}


def get_ctype_name(type_):
    if isinstance(type_, (ir.ArrayType, ir.ArrayArg)):
        # numpy array type
        # Todo: array arg needs to match a specific struct
        return "PyArrayObject*"
    else:
        dtype = npy_scalar_type_map.get(type_)
        if dtype is None:
            msg = f"Unsupported type {type_}"
            raise CompilerError(msg)
        return dtype


def get_ctype_tag(type_):
    # Todo: add something to distinguish uniform, by dim, sliding window
    if isinstance(type_, (ir.ArrayType, ir.ArrayArg)):
        ndims = type_.ndims
        dtype = scalar_type_mangling.get(type_.dtype)
        if dtype is None:
            msg = f"Array has unrecognized element type: {type_.dtype}"
            raise CompilerError(msg)
        tag = f"a{ndims}{dtype}"
    else:
        tag = scalar_type_mangling.get(type_)
        if tag is None:
            msg = f"Scalar has unrecognized type: {type_}"
            raise CompilerError(msg)
    return tag


# This assumes that C99 floats are 32 bits and C doubles are 64 bits.
# Numpy seems to depend on this anyway and probably won't compile on
# platforms where this does not hold.


def mangle_func_name(basename, arg_types):
    tag = "".join(get_ctype_tag(t) for t in arg_types)
    mangled_name = f"{basename}_{tag}"
    return mangled_name


def make_func_sig(func: ir.Function, syms: symbol_table):
    # maintain argument order
    args = tuple(syms.lookup(arg) for arg in func.args)
    arg_names = []
    for arg in func.args:
        sym = syms.lookup(arg)
        name = sym.name
        type_ = get_ctype_name(sym.type_)
        arg_names.append(f"{type_} {name}")
    types = [syms.check_type(arg) for arg in func.args]
    basename = extract_name(func.name)
    mangled_name = mangle_func_name(basename, types)
    formatted_args = []
    for type_, arg in zip(types, args):
        type_str = get_ctype_name(type_)
        formatted_args.append(type_str)
    return mangled_name, formatted_args


class Emitter:

    def __init__(self, path, indent="    ", max_line_width=70):
        self._indent = ""
        self.path = path
        self.single_indent = indent
        self.max_line_width = max_line_width
        self.line_buffer = []

    @property
    def indent(self):
        return self._indent

    @indent.setter
    def indent(self, level):
        self._indent = level

    def blank_lines(self, count=1):
        for c in range(count):
            self.line_buffer.append("")

    @contextmanager
    def ifdef_directive(self, cond):
        line = f"#ifdef {cond}"
        self.print_line(line)
        yield
        self.print_line("#endif")

    @contextmanager
    def ifndef_directive(self, cond):
        line = f"#ifndef {cond}"
        self.print_line(line)
        yield
        self.print_line("#endif")

    @contextmanager
    def indented(self):
        self._indent = f"{self._indent}{self.single_indent}"
        yield
        self._indent = self._indent[:-len(self.single_indent)]

    @contextmanager
    def curly_braces(self, line=None, semicolon=False):
        if line is not None:
            line = f"{line} {'{'}"
        else:
            line = "{"
        self.print_line(line)
        with self.indented():
            yield
        if semicolon:
            self.print_line("};")
        else:
            self.print_line("}")

    @contextmanager
    def flush_on_exit(self):
        yield
        self.flush()

    def print_line(self, line):
        lines = textwrap.wrap(initial_indent=self._indent, subsequent_indent=self._indent, text=line)
        self.line_buffer.extend(lines)

    def flush(self):
        if self.line_buffer:
            output_gen = "\n".join(line for line in self.line_buffer)
            pathlib.Path(self.path).write_text(output_gen)
            self.line_buffer.clear()


def ret_if_errored(printer: Emitter, expr=f"PyErr_Occurred()", retval="NULL"):
    cond = f"if ({expr})"
    with printer.curly_braces(line=cond):
        if retval:
            printer.print_line(f"return {retval};")
        else:
            printer.print_line("return;")


def ret_if_null(printer: Emitter, expr):
    cond = f"{expr} == NULL"
    ret_if_errored(printer, expr=cond)


def else_is_elif(stmt: ir.IfElse):
    if len(stmt.else_branch) == 1:
        if isinstance(stmt.else_branch[0], ir.IfElse):
            return True
    return False


# We have 3 layers here..
# Arg parsing layer,
# entry execution layer
# vectorized call layer
#
#
# Vectorization applied aggressively under a very conservative model.
# At function scope, we need x -> f(x)
# at loop scope,
# for index, values in enumerate(zip(...)):
#     out[index] = f(values)


# Todo: need utilities to check for cases where output type does not match input operand types
#       C99 has totally different type promotion rules, so it's better to break expressions
#       and add indicators to determine exact cast types.

# Todo: we need lowering for overflow checked arithmetic. It might be better to provide most of this
#       via a header.


class Formatter(ExpressionVisitor):

    def braced(self, node):
        expr = self.visit(node)
        return f"{expr}"

    def parenthesized(self, node):
        expr = self.visit(node)
        return f"({expr})"

    def __init__(self, symbols):
        self.infer_type = ExprTypeInfer(symbols)

    def __call__(self, node):
        return self.visit(node)

    @singledispatchmethod
    def visit(self, node):
        msg = f"No method to convert node: {node} to C99 code."
        raise NotImplementedError(msg)

    @visit.register
    def _(self, node: ir.MinReduction):
        subexprs = node.subexprs
        first_term = self.visit(next(subexprs))
        nested = first_term
        # turn this into a nesting of single min ops
        for subexpr in subexprs:
            subexpr = self.visit(subexpr)
            nested = f"min({nested}, {subexpr})"
        # if we are left with a one term expression,
        # then this is either an error or should have been tagged as an array reduction
        if nested == first_term:
            msg = f"single term reduction is ambiguous and should not have made it to this point {first_term}"
            raise RuntimeError(msg)
        return nested

    @visit.register
    def _(self, node: ir.MaxReduction):
        subexprs = node.subexprs
        first_term = self.visit(next(subexprs))
        nested = first_term
        # turn this into a nesting of single min ops
        for subexpr in subexprs:
            subexpr = self.visit(subexpr)
            nested = f"max({nested}, {subexpr})"
        # if we are left with a one term expression,
        # then this is either an error or should have been tagged as an array reduction
        if nested == first_term:
            msg = f"single term reduction is ambiguous and should not have made it to this point {first_term}"
            raise RuntimeError(msg)
        return nested

    @visit.register
    def _(self, node: ir.Length):
        if isinstance(node.operand, ir.NameRef):
            arr = self.visit(node.operand)
            return f"PyArray_DIM({arr}, 0)"
        else:
            msg = f"No method to convert {node.operand} to length."

    @visit.register
    def _(self, node: ir.SingleDimRef):
        dim = self.visit(node.dim)
        arr = self.visit(node.base)
        return f"PyArray_DIM({arr}, {dim})"

    @visit.register
    def _(self, node: ir.Max):
        left, right = node.subexprs
        left = self.visit(left)
        right = self.visit(right)
        return f"max({left}, {right})"

    @visit.register
    def _(self, node: ir.Min):
        left, right = node.subexprs
        left = self.visit(left)
        right = self.visit(right)
        return f"min({left}, {right})"

    @visit.register
    def _(self, node: ir.Select):
        predicate, on_true, on_false = (self.visit(subexpr)
                                        for subexpr in node.subexprs)
        expr = f"select({on_true}, {on_false}, {predicate})"
        return expr

    @visit.register
    def _(self, node: ir.Constant):
        return str(node.value)

    @visit.register
    def _(self, node: ir.StringConst):
        return f"\"{node.value}\""

    @visit.register
    def _(self, node: ir.BinOp):
        op = ir.binop_ops[type(node)]
        # Todo: We probably need to split inplace ops to a different path given the number of
        #  edge cases that arise when converting to C99 and including simd types.
        #  should we allow sleef for simd pow?
        #  pow is promoted to a call here either way, so it's already protected by parentheses.
        if op in ("**", "**="):
            left, right = (self.visit(subexpr) for subexpr in node.subexprs)
            expr = f"pow({left}, {right})"
        else:
            op_ordering = binop_ordering[op]
            left, right = (self.parenthesized(subexpr)
                           if ((isinstance(subexpr, ir.BinOp)
                                and op_ordering < binop_ordering[ir.binop_ops[type(subexpr)]])
                               or isinstance(subexpr, ir.CompareOp))
                           else self.visit(subexpr)
                           for subexpr in node.subexprs)
            expr = f"{left} {op} {right}"
        return expr

    @visit.register
    def _(self, node: ir.CompareOp):
        left, right = (self.parenthesized(subexpr)
                       if isinstance(subexpr, (ir.BoolOp, ir.CompareOp))
                       else self.visit(subexpr)
                       for subexpr in node.subexprs)
        op = ir.compare_ops[type(node)]
        expr = f"{left} {op} {right}"
        return expr

    @visit.register
    def _(self, node: ir.AND):
        expr = " && ".join(self.parenthesized(operand)
                           if isinstance(operand, (ir.AND, ir.OR))
                           else self.visit(operand)
                           for operand in node.subexprs)
        return expr

    @visit.register
    def _(self, node: ir.OR):
        expr = " || ".join(self.visit(operand)
                           for operand in node.subexprs)
        return expr

    @visit.register
    def _(self, node: ir.NOT):
        formatted = (self.parenthesized(node.operand)
                     if isinstance(node.operand, (ir.AND, ir.OR))
                     else self.visit(node.operand))
        expr = f"!{formatted}"
        return expr

    @visit.register
    def _(self, node: ir.TRUTH):
        formatted = (self.parenthesized(node.operand)
                     if isinstance(node.operand, ir.Subscript)
                     else self.visit(node.operand))
        formatted = f"(bool){formatted}"
        return formatted

    @visit.register
    def _(self, node: ir.NameRef):
        expr = node.name
        return expr

    @visit.register
    def _(self, node: ir.Call):
        func_name = self.visit(node.func)
        args = ", ".join(self.visit(arg) for arg in node.args)
        func = f"{func_name}({args})"
        return func

    @visit.register
    def _(self, node: ir.Subscript):
        t = self.infer_type(node)
        if not isinstance(t, ir.ScalarType):
            # no support for generating a sub-array ref here
            pf = pretty_formatter()
            msg = f"The code printer requires that subscripts reduce to scalar references. " \
                  f"Received '{pf(node)}' with type '{pf(t)}'."
            raise NotImplementedError(msg)
        base_type = self.infer_type(node.value)
        if not isinstance(base_type, ir.ArrayType):
            raise NotImplementedError()
        if not (1 <= base_type.ndims <= 4):
            raise NotImplementedError()
        # get c cast type
        cast_type = get_ctype_name(t)
        # make dereference
        if isinstance(node.index, ir.Tuple):
            index = ", ".join(self.visit(i) for i in node.index.subexprs)
        else:
            index = self.visit(node.index)
        # generate cast pointer at the appropriate offset
        s = f"*(({cast_type}*) PyArray_GETPTR{base_type.ndims}({index}))"
        return s

    @visit.register
    def _(self, node: ir.USUB):
        operand = (self.parenthesized(node.operand)
                   if (isinstance(node.operand, ir.BinOp) and not isinstance(node.operand, ir.POW))
                   else self.visit(node.operand))
        return f"-{operand}"

    @visit.register
    def _(self, node: ir.UNOT):
        operand = (self.parenthesized(node.operand)
                   if (isinstance(node.operand, ir.BinOp) and not isinstance(node.operand, ir.POW))
                   else self.visit(node.operand))
        expr = f"~{operand}"
        return expr


# need a header file generator..


class BoilerplateWriter:

    # This is meant to be controlled by a codegen driver,
    # which manages opening/closing of a real or virtual destination file.

    def __init__(self, emitter, modname):
        self.printer = emitter
        self.modname = modname

    def print_sys_header(self, name):
        s = f"#include<{name}>"
        self.printer.print_line(s)

    def print_user_header(self, name):
        s = f"#include \"{name}\""
        self.printer.print_line(s)

    def gen_source_top(self, sys_headers=(), user_headers=()):
        self.printer.print_line("#define PY_SSIZE_T_CLEAN")
        self.print_sys_header("Python.h")
        for h in sys_headers:
            self.print_sys_header(h)
        for h in user_headers:
            self.print_user_header(h)

    def gen_module_init(self):
        if self.modname == "mod":
            raise CompilerError("mod is treated as a reserved name.")
        line = f"PyMODINIT_FUNC PyInit_{self.modname}(void)"
        with self.printer.curly_braces(line=line):
            self.printer.print_line("import_array();")
            self.printer.print_line("import_ufunc();")
            self.printer.print_line(f"PyObject* mod = PyModule_Create(&{self.modname});")
            line = "if(mod == NULL)"
            with self.printer.curly_braces(line=line):
                self.printer.print_line("return NULL;")
            self.printer.print_line("return mod;")

    def gen_method_table(self, funcs):
        # no keyword support..
        line = f"static PyMethodDef {self.modname}Methods[] ="
        with self.printer.curly_braces(line=line, semicolon=True):
            for index, (base, mangled) in enumerate(funcs):
                line = f"{'{'}\"{base}\", {mangled}, METH_VARARGS, NULL{'}'},"
                self.printer.print_line(line)
            # sentinel ending entry
            self.printer.print_line("{NULL, NULL, 0, NULL}")

    def gen_module_def(self):
        line = f"static PyModuleDef {self.modname} ="
        with self.printer.curly_braces(line=line, semicolon=True):
            self.printer.print_line("PyModuleDef_HEAD_INIT,")
            self.printer.print_line(f"{self.modname},")
            self.printer.print_line("NULL,")  # no module docstring support
            self.printer.print_line("-1,")  # no support for per interpreter state tracking
            self.printer.print_line(f"{self.modname}Methods")  # method table


class ModuleCodeGen(StmtVisitor):
    array_decl_type_name = "PyArrayObject"

    def __init__(self, modname: str, printer):
        self.modname = modname
        self.printer = printer
        self._declared = None
        self.infer_type = None
        self.format = None

    @contextmanager
    def function_context(self, symbols):
        self.symbols = symbols
        self.format = Formatter(symbols)
        self.infer_type = ExprTypeInfer(symbols)
        self._declared = set()
        yield
        self.symbols = None
        self.format = None
        self.infer_type = None
        self._declared = None

    def declare(self, name):
        name = ir.NameRef(name)
        if isinstance(name, ir.NameRef):
            self._declared.add(name)

    def declared(self, name):
        return name in self._declared

    def __call__(self, node, symbols: symbol_table):
        with self.function_context(symbols):
            for arg in symbols.arguments:
                self.declare(arg.name)
            self.visit(node)

    @singledispatchmethod
    def visit(self, node):
        msg = f"no ModuleCodeGen visitor for node {node} of type {type(node)}"
        raise NotImplementedError(msg)

    @visit.register
    def _(self, node: list):
        for stmt in node:
            self.visit(stmt)

    @visit.register
    def _(self, node: ir.Assign):
        # check types
        rhs_type = self.infer_type(node.value)
        lhs_type = self.infer_type(node.target)
        if lhs_type != rhs_type:
            msg = f"Cannot cast type {rhs_type} to type {lhs_type} on assignment: line {node.pos.line_begin}."
            raise CompilerError(msg)
        # already declared
        target = self.format(node.target)
        value = self.format(node.value)
        stmt = f"{target} = {value};"
        self.printer.print_line(stmt)

    def visit_elif(self, node: ir.IfElse):
        test = f"else if({self.format(node.test)})"
        with self.printer.curly_braces(line=test):
            self.visit(node.if_branch)
        if else_is_elif(node):
            self.visit_elif(node.else_branch[0])
        elif node.else_branch:
            with self.printer.curly_braces(line="else"):
                self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.IfElse):
        test = self.format(node.test)
        # with self.scoped("if", test):
        with self.printer.curly_braces(line=f"if({test})"):
            self.visit(node.if_branch)
        if else_is_elif(node):
            self.visit_elif(node.else_branch[0])
        elif node.else_branch:
            with self.printer.curly_braces(line="else"):
                self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.ForLoop):
        # Printing is more restrictive, check that
        # we have a supported loop structure
        assert isinstance(node.target, ir.NameRef)
        assert isinstance(node.iterable, (ir.AffineSeq, ir.Reversed))
        # check for unit step
        target = self.format(node.target)
        assert isinstance(node.target, ir.NameRef)
        type_ = get_ctype_name(self.symbols.check_type(node.target))
        # check whether we can use ++ op
        # insufficient support for reversed() thus far.
        if node.iterable.step == ir.One:
            step_expr = f"++{target}"
        else:
            increm_by = self.format(node.iterable.step)
            step_expr = f"{target} += {increm_by}"
        start = self.format(node.iterable.start)
        # forming the expression here avoids missing parentheses
        stop_cond = self.format(node.iterable.stop)
        cond = f"for({type_} {target} = {start}; {target} < {stop_cond}; {step_expr})"
        with self.printer.curly_braces(line=cond):
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.WhileLoop):
        cond = self.format(node.test)
        with self.printer.curly_braces(line=f"while({cond})"):
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.Break):
        self.printer.print_line("break;")

    @visit.register
    def _(self, node: ir.Continue):
        self.printer.print_line("continue;")

    @visit.register
    def _(self, node: ir.Assign):
        target = self.format(node.target)
        if isinstance(node.value, ir.ArrayInitializer):
            # get array type  for target
            # no support for assigning array allocation to NameRef
            if not isinstance(node.target, ir.NameRef):
                msg = "Array initialization target must be a NameRef"
                raise CompilerError(msg)
            target_type = self.symbols.check_type(node.target)
            # check that these match, this could be hoisted prior to codegen
            assert target_type.dtype == node.value.dtype
            assert target_type.ndims == node.value.shape
            # assume array types are always pre-declared
            base_name = extract_name(node.target)
            dims_name = self.symbols.make_unique_name_like(f"{base_name}_dims")
            ndims = node.value.shape
            array_init = node.value
            init_dims_stmt = f"{dims_name}[{ndims}] = {'{'}{', '.join(self.format(d) for d in array_init.shape.subexprs)}{'}'};"
            self.printer.print_line(init_dims_stmt)
            dtype = npy_dtype_codes.get(target_type.dtype)
            call_name = "PyArray_Zeros" if isinstance(target, ir.Zeros) else "PyArray_Empty"
            # Todo: this needs to be paired with access funcs to grab 1D arrays
            stmt = f"{self.array_decl_type_name}* {base_name} = {call_name}({ndims}, &{dims_name}, {dtype}, false);"
            self.printer.print_line(stmt)
        else:
            value = self.format(node.value)
            op = "="
            line = f"{target} {op} {value};"
            self.printer.print_line(line)

    @visit.register
    def _(self, node: ir.SingleExpr):
        expr = self.format(node.expr)
        line = f"{expr};"
        self.printer.print_line(line)

    @visit.register
    def _(self, node: ir.Return):
        # Todo: These need to be split to a degree, depending on whether
        # this is called from Python or C. In reality, for calls from C,
        # we're probably pushing an out. For Python we need a PyObject*.
        if node.value is not None:
            value = self.format(node.value)
            self.printer.print_line(f"return {value};")
        else:
            self.printer.print_line("return;")


class ReturnCheck(StmtVisitor):

    def __init__(self):
        self.return_values = None

    def __call__(self, node):
        self.return_values = set()
        values = self.return_values
        self.return_values = None
        return values

    def visit(self, node):
        if isinstance(node, ir.Return):
            self.return_values.add(node.value)
        else:
            super().visit(node)


def check_return_type(node: ir.Function, symbols: symbol_table):
    rc = ReturnCheck()
    values = rc(node)
    possible_types = {symbols.check_type(v) for v in values}
    # should check if all paths terminate..
    # eg either ends in a return statement or every branch has a return statement
    if len(possible_types) > 1:
        msg = f"Function {extract_name(node)} has more than one return type, received {possible_types}."
        raise CompilerError(msg)
    elif len(possible_types) == 1:
        return_type = possible_types.pop()
    else:
        return_type = None  # coerce to void at C level outside wrapper
    return return_type


def make_single_iter_func(func: ir.Function, symbols: symbol_table, printer: Emitter):
    mangled_name, arg_types = make_func_sig(func, symbols)
    # Generate python wrapper
    return_type = "void"  # temporary standin..
    args = []
    for arg in func.args:
        arg_name = extract_name(arg)
        t = symbols.check_type(arg_name)
        if isinstance(t, ir.ArrayType):
            arg_str = f"PyArrayObject* {arg_name}"
        else:
            ctype = npy_scalar_type_map[t]
            arg_str = f"{ctype} {arg_name}"
        args.append(arg_str)
    args_str = ", ".join(args)
    with printer.curly_braces(line=f"{return_type} {mangled_name}({args_str})"):
        # print decls
        by_type = defaultdict(set)
        for sym in symbols.all_locals:
            if sym.is_arg:
                continue
            base_type = symbols.check_type(sym.name)
            ctype = get_ctype_name(base_type)
            by_type[ctype].add(sym.name)
        for ctype_, names in by_type.items():
            decls = ", ".join(n for n in names)
            printer.print_line(f"{ctype_} {decls};")
        builder = ModuleCodeGen(modname="", printer=printer)
        builder(func.body, symbols)


def make_py_wrapper(modname: str, func: ir.Function, symbols: symbol_table, printer: Emitter):
    # reserve arg name
    arg_name = extract_name(symbols.make_unique_name_like("args", ir.CObjectType(is_array=False)))
    self_name = extract_name(symbols.make_unique_name_like("self", ir.CObjectType(is_array=False)))

    wrapper_name = f"{modname}_{func.name}"
    local_names = symbol_table("temporary", {})
    header = f"static PyObject* {wrapper_name}(PyObject* {self_name}, PyObject* {arg_name})"
    with printer.curly_braces(line=header):
        # declare generic objects by position
        # may want to mangle..
        arg_to_temp_name = {}
        arg_var_refs = []
        for index, arg in enumerate(func.args):
            t = symbols.check_type(arg)
            name = local_names.make_unique_name_like("arg", t)
            name = extract_name(name)
            arg_init = f"PyObject* {name} = NULL;"
            printer.print_line(arg_init)
            arg_to_temp_name[extract_name(arg)] = name
            arg_var_refs.append(f"&{name}")

        object_str = "".join(itertools.repeat("O", len(func.args)))

        # need object refs from initial parse
        arg_var_refs = ", ".join(arg_var_refs)
        cond = f'!PyArg_ParseTuple(args, "{object_str}", {arg_var_refs})'
        ret_if_errored(printer, expr=cond)

        for index, arg in enumerate(func.args):
            t = symbols.check_type(arg)
            if isinstance(t, ir.ArrayType):
                np_dtype = ir.by_ir_type[t.dtype]
                type_num = npy_c_type_codes[np_dtype]
                name = extract_name(arg)
                obj_name = arg_to_temp_name[name]
                expected_ndims = t.ndims
                init = f"PyArrayObject* {name} = unwrap_array({obj_name}, {type_num}, {expected_ndims});"
                printer.print_line(init)

            elif isinstance(t, ir.ScalarType):
                ctype = scalar_type_map[t]
                unwrapper = npy_scalar_unwrap_funcs[t]
                name = extract_name(arg)
                temp_name = arg_to_temp_name[name]
                line = f"{ctype} {name} = {unwrapper}({temp_name});"
                printer.print_line(line)

            else:
                msg = f"Unrecognized argument type {t}."
                raise CompilerError(msg)

        # now generate a call to the unwrapped version
        arg_names = (extract_name(n) for n in func.args)
        # arg_str = ", ".join(arg_names)
        mangled_name, formatted_args = make_func_sig(func, symbols)
        arg_str = ", ".join(fa for fa in formatted_args)
        stmt = f"return {mangled_name}({arg_str});"
        printer.print_line(stmt)


# This needs to distinguish interpreter facing from internal

def codegen(build_dir, funcs: Iterable[ir.Function], symbols: Dict[str, symbol_table], modname):
    file_path = pathlib.Path(build_dir).joinpath(f"{modname}Module.c")
    printer = Emitter(file_path)
    bp_gen = BoilerplateWriter(printer, modname)
    sys_headers = ("ndarraytypes.h.",)
    methods = []
    # get return type
    with printer.flush_on_exit():
        bp_gen.gen_source_top(sys_headers)
        # These are just way too difficult to read as inline ternary ops, and their precedence
        # flips completely when used with intrinsics, making this seem like a sensible option
        with printer.ifndef_directive("min"):
            printer.print_line("#define min(x,y) (x < y ? x : y)")
        with printer.ifndef_directive("max"):
            printer.print_line("#define max(x,y) (x > y ? x : y)")
        with printer.ifndef_directive("select"):
            printer.print_line("#def select(x, y, predicate) (predicate ? x : y)")
        printer.blank_lines(count=2)
        bp_gen.gen_module_init()
        printer.blank_lines(count=2)

        for func in funcs:
            func_symbols = symbols.get(func.name)
            make_single_iter_func(func, func_symbols, printer)
            printer.blank_lines(count=2)

        for func in funcs:
            func_symbols = symbols.get(func.name)
            func_name = extract_name(func)
            meth = (func_name, func_name)
            methods.append(meth)
            make_py_wrapper(modname, func, func_symbols, printer)
            printer.blank_lines(count=2)

        # Generate python wrapper
        bp_gen.gen_method_table(methods)
        printer.blank_lines(count=2)
        bp_gen.gen_module_def()
