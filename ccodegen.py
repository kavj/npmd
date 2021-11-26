import pathlib
import textwrap
import pathlib

from contextlib import contextmanager
from functools import singledispatchmethod

import ir

import type_resolution as tr

from errors import CompilerError
from lowering import ReductionToSelect
from pretty_printing import binop_ordering
from symbol_table import symbol, symbol_table
from utils import extract_name, wrap_input
from visitor import ExpressionTransformer, StmtVisitor

"""

most of the transforms are done elsewhere
This should load the appropriate ISA for simd ops and

This still needs a customized visitor to generate statements.
At this point, statements are serialized to the level expected by C code.

"""

std_include_headers = ["Python.h", "numpy/arrayobject.h", "stdint.h", "stdbool.h", "math.h"]

# initial function name mangling abi
# scalar types use a single letter + bit count
# This is meant to disambiguate multiple lowerings based on arguments.

scalar_type_mangling = {
    tr.Int32: "i32",
    tr.Int64: "i64",
    tr.Float32: "f32",
    tr.Float64: "f64",
    tr.Predicate32: "p32",
    tr.Predicate64: "p64",
    tr.FPredicate32: "q32",
    tr.FPredicate64: "q64",
    tr.BoolType: "b8"
}

scalar_type_map = {tr.Int32: "int32_t",
                   tr.Int64: "int64_t",
                   tr.Float32: "float",
                   tr.Float64: "double",
                   tr.Predicate32: "bool",
                   tr.Predicate64: "bool",
                   tr.BoolType: "bool"}


def get_ctype_name(type_):
    if isinstance(type_, (ir.ArrayType, ir.ArrayArg)):
        # numpy array type
        # Todo: array arg needs to match a specific struct
        return "PyArrayObject"
    else:
        dtype = scalar_type_map.get(type_)
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


def mangle_func_name(basename, arg_types, modname):
    tag = "".join(get_ctype_tag(t) for t in arg_types)
    mangled_name = f"{basename}_{tag}"
    return mangled_name


def make_func_sig(func: ir.Function, syms: symbol_table, modname):
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
    mangled_name = mangle_func_name(basename, types, modname)
    formatted_args = []
    for type_, arg in zip(types, args):
        type_str = get_ctype_name(type_)
        if isinstance(arg.type_, (ir.ArrayType, ir.ArrayArg)):
            type_str = f"{type_str}*"
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
    def curly_braces(self, semicolon=False):
        self.print_line("{")
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


def else_is_elif(stmt: ir.IfElse):
    if len(stmt.else_branch) == 1:
        if isinstance(stmt.else_branch[0], ir.IfElse):
            return True
    return False


# Todo: need utilities to check for cases where output type does not match input operand types
#       C99 has totally different type promotion rules, so it's better to break expressions
#       and add indicators to determine exact cast types.

# Todo: we need lowering for overflow checked arithmetic. It might be better to provide most of this
#       via a header.


class Formatter(ExpressionTransformer):

    def braced(self, node):
        expr = self.visit(node)
        return f"{expr}"

    def parenthesized(self, node):
        expr = self.visit(node)
        return f"({expr})"

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
    def _(self, node: ir.BoolConst):
        return str(node.value)

    @visit.register
    def _(self, node: ir.IntConst):
        return str(node.value)

    @visit.register
    def _(self, node: ir.FloatConst):
        return str(node.value)

    @visit.register
    def _(self, node: ir.StringConst):
        return f"\"{node.value}\""

    @visit.register
    def _(self, node: ir.BinOp):
        op = node.op
        # Todo: We probably need to split inplace ops to a different path given the number of
        #  edge cases that arise when converting to C99 and including simd types.
        #  should we allow sleef for simd pow?
        #  pow is promoted to a call here either way, so it's already protected by parentheses.
        if op in ("**", "**="):
            left, right = (self.visit(subexpr) for subexpr in node.subexprs)
            expr = f"pow({left}, {right})"
        elif node.in_place:
            left, right = (self.visit(subexpr) for subexpr in node.subexprs)
            expr = f"{left} {op} {right}"
        else:
            op_ordering = binop_ordering[op]
            left, right = (self.parenthesized(subexpr)
                           if ((isinstance(subexpr, ir.BinOp) and op_ordering < binop_ordering[subexpr.op])
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
        expr = f"{left} {node.op} {right}"
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
        s = f"{self.visit(node.value)}[{self.visit(node.slice)}]"
        return s

    @visit.register
    def _(self, node: ir.UnaryOp):
        op = node.op
        operand = (self.parenthesized(node.operand)
                   if (isinstance(node.operand, ir.BinOp) and node.operand.op != "**")
                   else self.visit(node.operand))
        expr = f"{op}{operand}"
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
        self.printer.print_line(f"PyMODINIT_FUNC PyInit_{self.modname}(void)")
        with self.printer.curly_braces():
            self.printer.print_line("import_array();")
            self.printer.print_line("import_ufunc();")
            self.printer.print_line(f"PyObject* mod = PyModule_Create(&{self.modname});")
            self.printer.print_line("if(mod == NULL)")
            with self.printer.curly_braces():
                self.printer.print_line("return NULL;")
            self.printer.print_line("return mod;")

    def gen_method_table(self, funcs):
        # no keyword support..
        self.printer.print_line(f"static PyMethodDef {self.modname}Methods[] =")
        with self.printer.curly_braces(semicolon=True):
            last = len(funcs) - 1
            for index, (base, mangled) in enumerate(funcs):
                line = f"{'{'}\"{base}\", {mangled}, METH_VARARGS, NULL{'}'},"
                self.printer.print_line(line)
            # sentinel ending entry
            self.printer.print_line("{NULL, NULL, 0, NULL}")

    def gen_module_def(self):
        self.printer.print_line(f"static PyModuleDef {self.modname} =")
        with self.printer.curly_braces(semicolon=True):
            self.printer.print_line("PyModuleDef_HEAD_INIT,")
            self.printer.print_line(f"{self.modname},")
            self.printer.print_line("NULL,")  # no module docstring support
            self.printer.print_line("-1,")  # no support for per interpreter state tracking
            self.printer.print_line(f"{self.modname}Methods")  # method table

    def convert_py_obj_to_array(self, obj, target):
        # unwind result of PyArg_ParseTuple
        # need a simple call to validate this, too much to do inline

        pass

    def gen_decref(self, target):
        #
        pass

    def error_if_true(self, cond):
        pass

    def error_if_false(self, cond):
        pass

    def check_flags(self, target):
        # This should be a call, not inline
        pass

    def get_ptr(self, target):
        pass


class ModuleCodeGen(StmtVisitor):

    def __init__(self, modname: str, printer):
        self.modname = modname
        self.format = Formatter()
        self.printer = printer
        self.symbols = None

    @contextmanager
    def function_context(self, symbols):
        assert self.symbols is None
        self.symbols = symbols
        self._declared = set()
        yield
        self.symbols = None

    def declare(self, name):
        name = wrap_input(name)
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
        rhs_type = self.check_type(node.value)
        lhs_type = self.check_type(node.target)
        if lhs_type != rhs_type:
            msg = f"Cannot cast type {rhs_type} to type {lhs_type} on assignment: line {node.pos.line_begin}."
            raise CompilerError(msg)
        if node.in_place:
            if isinstance(node.target, ir.NameRef):
                assert node.target in self.declared
            # already declared or compiler error
            stmt = f"{self.visit(node.value)};"
        elif not self.declared:
            ctype_name = get_ctype_name(lhs_type)
            target = f"{ctype_name} {self.format(node.target)}"
            value = self.format(node.value)
            stmt = f"{target} = {value};"
        else:
            # already declared
            target = self.format(node.target)
            value = self.format(node.value)
            stmt = f"{target} = {value};"
        self.printer.print_line(stmt)

    def visit_elif(self, node: ir.IfElse):
        test = f"else if({self.format(node.test)})"
        self.printer.print_line(test)
        with self.printer.curly_braces():
            self.visit(node.if_branch)
        if else_is_elif(node):
            self.visit_elif(node.else_branch[0])
        elif node.else_branch:
            self.printer.print_line("else")
            with self.printer.curly_braces():
                self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.IfElse):
        test = self.format(node.test)
        # with self.scoped("if", test):
        self.printer.print_line(f"if({test})")
        with self.printer.curly_braces():
            self.visit(node.if_branch)
        if else_is_elif(node):
            self.visit_elif(node.else_branch[0])
        elif node.else_branch:
            self.printer.print_line("else")
            with self.printer.curly_braces():
                self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.ForLoop):
        # Printing is more restrictive, check that
        # we have a supported loop structure
        assert isinstance(node.target, ir.NameRef)
        assert isinstance(node.iterable, (ir.AffineSeq, ir.Reversed))
        # check for unit step
        target = self.format(node.target)
        if isinstance(node.target, ir.NameRef) and not self.declared(node.target):
            self.declare(node.target)
            type_ = get_ctype_name(self.symbols.check_type(node.target))
            decl = f"{type_} {target}"
        else:
            decl = target
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
        cond = f"for({decl} = {start}; {target} < {stop_cond}; {step_expr})"
        self.printer.print_line(cond)
        with self.printer.curly_braces():
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.WhileLoop):
        cond = self.format(node.test)
        self.printer.print_line(f"while({cond})")
        with self.printer.curly_braces():
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.Break):
        self.printer.print_line("break;")

    @visit.register
    def _(self, node: ir.Continue):
        self.printer.print_line("continue;")

    @visit.register
    def _(self, node: ir.Assign):
        if isinstance(node.target, ir.NameRef) and not self.declared(node.target):
            type_ = get_ctype_name(self.symbols.check_type(node.target))
            target = f"{type_} {self.format(node.target)}"
        else:
            target = self.format(node.target)
        if isinstance(node.value, ir.ArrayInit):
            # get array type for target
            # no support for assigning array allocation to NameRef
            if not isinstance(node.target, ir.NameRef):
                raise CompilerError
            target_type = self.symbols.check_type(node.target)
            # check that these match, this could be hoisted prior to codegen
            assert target_type.dtype == node.value.dtype
            assert target_type.ndims == node.value.ndims
            # assume array types are always pre-declared
            dims = [self.visit(dim) for dim in node.value.dims]
            elem_count = " * ".join(d for d in dims)
            c_dtype = get_ctype_name(target_type.dtype)
            # Todo: replace with more general allocator
            alloc = f"malloc({elem_count} * sizeof({c_dtype}))"
            stmt = f"{target}.data = {alloc};"
            self.printer.print_line(stmt)
            for i, dim in enumerate(dims):
                self.printer.print_line(f"{target}.d{i} = {dim};")
        else:
            value = self.format(node.value)
            if node.in_place:
                assert isinstance(node.value, ir.BinOp)
                assign_op = node.value.op
            else:
                assign_op = "="
            line = f"{target} {assign_op} {value};"
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


def format_string_from_type():
    # get workable C type
    pass


def verify_array_type():
    # insert runtime code to check that an array object is the type specified
    pass


def make_py_wrapper(modname, func, symbols, sig, printer):
    mangled_name, arg_types = sig
    wrapper_name = f"{modname}_{func.name}"
    # use standard int when aiming for 32 bits,
    # coerce as necessary
    type_map = {tr.Int32: "i", tr.Int64: "L", tr.Float32: "f", tr.Float64: "d", ir.ArrayType: "O"}

    # get type string
    format_strs = []
    for arg in func.args:
        type_ = symbols.check_type(arg)
        if isinstance(type_, ir.ArrayType):
            format_strs.append("O")
        else:
            format_strs.append(type_map[type_])
    decode = "".join(fs for fs in format_strs)
    with printer.curly_braces():
        header = f"PyObject* {wrapper_name}(PyObject* self, PyObject* args)"
        # arg_types = tuple((arg, symbols.check_type(arg)) for arg in func.args)
        # declare arguments
        decls = ()
        self.printer.print_line(decls)
        self.printer.print_line(f"if(PyArg_ParseTuple(args, {decode})) < 0)")
        with self.printer.curly_braces():
            self.printer.print_line("return NULL;")
        # need return type


# This needs to distinguish interpreter facing from internal

def codegen(build_dir, funcs, symbols, modname):
    file_path = pathlib.Path(build_dir).joinpath(f"{modname}Module.c")
    printer = Emitter(file_path)
    mod_builder = ModuleCodeGen(modname, printer)
    bp_gen = BoilerplateWriter(printer, modname)
    sys_headers = ("ndarraytypes.h.",)
    func_lookup = {}
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
            basename = func.name
            func_symbols = symbols.get(basename)
            mangled_name, arg_types = make_func_sig(func, func_symbols, modname)
            # return_type = get_return_type(func)
            # Generate python wrapper
            wrapper_name = f"{modname}_{basename}"
            func_lookup[basename] = (mangled_name, arg_types)
            # return_type = get_ctype_name(return_type)
            return_type = "void"  # temporary standin..
            arg_names = (extract_name(arg) for arg in func.args)
            arg_str = ", ".join(f"{type_} {arg}" for (type_, arg) in zip(arg_types, arg_names))
            sig = f"{return_type} {mangled_name}({arg_str})"
            printer.print_line(sig)
            with printer.curly_braces():
                # print decls
                decls = ", ".join(f"{get_ctype_name(sym.type_)} {sym.name}"
                                  for sym in func_symbols.source_locals)
                if decls:
                    # skip if no local variables declared in source
                    printer.print_line(f"{decls};")
                mod_builder(func.body, func_symbols)
            methods.append((basename, mangled_name))
            printer.blank_lines(count=2)
        bp_gen.gen_method_table(methods)
        printer.blank_lines(count=2)
        bp_gen.gen_module_def()
