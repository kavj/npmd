import pathlib
import textwrap
import pathlib

from contextlib import contextmanager
from functools import singledispatchmethod

import ir

import type_resolution as tr

from errors import CompilerError
from pretty_printing import binop_ordering
from symbol_table import symbol, symbol_table
from utils import extract_name
from visitor import StmtVisitor

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
    mangled_name = f"{modname}{basename}_{tag}"
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
        if isinstance(arg.type_, ir.ArrayType):
            type_str = f"{type_str}*"
        formatted_args.append(type_str)
    return mangled_name, formatted_args


class Emitter:

    def __init__(self, path, indent="    ", max_line_width=70):
        self._indent = ""
        self.path = path
        self.single_indent = indent
        self.max_line_width = max_line_width
        self.line_formatter = textwrap.TextWrapper(tabsize=4, break_long_words=False, break_on_hyphens=False)
        self.line_buffer = []

    @property
    def indent(self):
        return self.line_formatter.initial_indent

    @indent.setter
    def indent(self, level):
        self._indent = level

    @property
    def indent_len(self):
        return len(self.line_formatter.initial_indent)

    def blank_lines(self, count=1):
        for c in range(count):
            self.line_buffer.append("")

    @contextmanager
    def ifdef_region(self, cond):
        line = f"#ifdef {cond}"
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
        indented_line = f"{self._indent}{line}"
        lines = self.line_formatter.wrap(indented_line)
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


class Formatter:

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
    def _(self, node: ir.SingleDimRef):
        pass

    @visit.register
    def _(self, node: ir.Max):
        assert len(node.values) == 2
        left_, right_ = node.values
        left = self.visit(left_)
        right = self.visit(right_)
        expr = f"{left} > {right} ? {left} : {right}"
        return expr

    @visit.register
    def _(self, node: ir.Min):
        assert len(node.values) == 2
        left_, right_ = node.values
        left = self.visit(left_)
        right = self.visit(right_)
        expr = f"{left} < {right} ? {left} : {right}"
        return expr

    @visit.register
    def _(self, node: ir.Select):
        test = self.parenthesized(node.test) if isinstance(node.test, ir.Select) else self.visit(node.test)
        on_true = self.parenthesized(node.on_true) if isinstance(node.on_true, ir.Select) \
            else self.visit(node.on_true)
        on_false = self.parenthesized(node.on_false) if isinstance(node.on_false, ir.Select) \
            else self.parenthesized(node.on_false)
        expr = f"{test} ? {on_true} : {on_false}"
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
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op
        # Todo: We probably need to split inplace ops to a different path given the number of
        #  edge cases that arise when converting to C99 and including simd types.
        #  should we allow sleef for simd pow?
        #  pow is promoted to a call here either way, so it's already protected by parentheses.
        if op in ("**", "**="):
            return f"pow({left}, {right})"
        elif not node.in_place:
            op_ordering = binop_ordering[op]
            if isinstance(node.left, ir.BinOp):
                if op_ordering < binop_ordering[node.left.op]:
                    left = self.parenthesized(left)
            elif isinstance(node.left, (ir.BoolOp, ir.CompareOp, ir.Select)):
                left = self.parenthesized(left)
            if isinstance(node.right, ir.BinOp):
                if op_ordering < binop_ordering[right.op]:
                    left = self.parenthesized(right)
            elif isinstance(node.right, (ir.BoolOp, ir.CompareOp, ir.Select)):
                right = self.parenthesized(right)
        expr = f"{left} {op} {right}"
        return expr

    @visit.register
    def _(self, node: ir.CompareOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.left, (ir.BoolOp, ir.CompareOp, ir.Select, ir.Tuple)):
            left = self.parenthesized(left)
        if isinstance(node.right, (ir.BoolOp, ir.CompareOp, ir.Select, ir.Tuple)):
            right = self.parenthesized(right)
        expr = f"{left} {node.op} {right}"
        return expr

    @visit.register
    def _(self, node: ir.AND):
        operands = []
        for operand in node.operands:
            formatted = self.visit(operand)
            if isinstance(operand, (ir.AND, ir.OR, ir.Select)):
                formatted = self.parenthesized(formatted)
            operands.append(formatted)
        expr = " && ".join(operand for operand in operands)
        return expr

    @visit.register
    def _(self, node: ir.OR):
        operands = []
        for operand in node.operands:
            formatted = self.visit(operand)
            if isinstance(operand, ir.Select):
                formatted = self.parenthesized(formatted)
            operands.append(formatted)
        expr = " || ".join(operand for operand in operands)
        return expr

    @visit.register
    def _(self, node: ir.NOT):
        formatted = self.visit(node.operand)
        if isinstance(node.operand, (ir.AND, ir.OR, ir.Select)):
            formatted = self.parenthesized(formatted)
        expr = f"!{formatted}"
        return expr

    @visit.register
    def _(self, node: ir.TRUTH):
        formatted = self.visit(node.operand)
        if node.constant:
            if not isinstance(node, ir.BoolConst):
                # We don't distinguish between bools and predicates here in
                # truth testing, since Python doesn't have any notion of
                # predicate types.
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
        operand = self.visit(node.operand)
        if isinstance(node.operand, ir.BinOp) and not node.operand.in_place:
            if node.operand.op != "**":
                operand = self.parenthesized(operand)
        elif isinstance(node.operand, (ir.UnaryOp, ir.BoolOp, ir.Select)):
            # if we have an unfolded double unary expression such as --,
            # '--expr' would be correct but it's visually jarring. Adding
            # unnecessary parentheses makes it '-(-expr)'.
            operand = self.parenthesized(operand)
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

    def gen_method_table(self, funcs):
        # no keyword support..
        self.printer.print_line(f"static PyMethodDef {self.modname}Methods[] =")
        with self.printer.curly_braces(semicolon=True):
            for base, mangled in funcs:
                line = f"\"{base}\", {mangled}, METH_VARARGS, NULL"
                with_braces = f"{'{'}{line}{'}'}"
                self.printer.print_line(with_braces)
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
        yield
        self.symbols = None

    def __call__(self, node, symbols: symbol_table):
        with self.function_context(symbols):
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
        target = self.format(node.target)
        value = self.format(node.value)
        if node.in_place:
            self.printer.print_line(f"{value};")
        else:
            # For now, assume C99 back end,
            # compliant with PEP 7
            stmt = f"{target} = {value};"
            self.printer.print_line(stmt)

    def visit_elif(self, node: ir.IfElse):
        test = self.format(node.test)
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
        # check whether we can use ++ op
        # insufficient support for reversed() thus far.
        if node.iterable.step == ir.One:
            step_expr = f"++{target}"
        else:
            increm_by = self.format(node.iterable.step)
            step_expr = f"{target} += {increm_by}"
        start = self.format(node.iterable.start)
        stop = self.format(node.iterable.stop)
        # Todo: should declare symbol here..
        cond = f"for({target} = {start}; {target} < {stop}; {step_expr})"
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
        printer.blank_lines(count=2)
        bp_gen.gen_module_init()
        printer.blank_lines(count=2)
        for func in funcs:
            basename = func.name
            func_symbols = symbols.get(basename)
            mangled_name, arg_seq = make_func_sig(func, func_symbols, modname)
            # return_type = get_return_type(func)
            func_lookup[basename] = (mangled_name, arg_seq)
            # return_type = get_ctype_name(return_type)
            return_type = "void"  # temporary standin..
            arg_str = ", ".join(arg for arg in arg_seq)
            sig = f"{return_type} {mangled_name}({arg_str})"
            printer.print_line(sig)
            with printer.curly_braces():
                mod_builder(func.body, symbols)
            methods.append((basename, mangled_name))
            printer.blank_lines(count=2)
        bp_gen.gen_method_table(methods)
        printer.blank_lines(count=2)
        bp_gen.gen_module_def()
