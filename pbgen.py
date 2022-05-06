import pathlib
import textwrap

import numpy as np

from contextlib import contextmanager
from functools import singledispatch, singledispatchmethod
from typing import Dict, List, Optional, Set

import ir

from errors import CompilerError
from reductions import simple_serialize_min_max
from symbol_table import symbol_table
from type_checks import TypeHelper, check_return_type
from utils import extract_name
from visitor import walk


npy_c_type_codes = {
    np.dtype("bool"): "NPY_BOOL",
    np.dtype("int8"): "NPY_INT8",
    np.dtype("uint8"): "NPY_UINT8",
    np.dtype("int32"): "NPY_INT32",
    np.dtype("int64"): "NPY_INT64",
    np.dtype("float32"): "NPY_FLOAT32",
    np.dtype("float64"): "NPY_FLOAT64",
}


npy_map = {np.dtype("bool"): "bool",
           np.dtype("int8"): "int8_t",
            np.dtype("uint8"): "uint8_t",
            np.dtype("int32"): "int32_t",
            np.dtype("int64"): "int64_t",
            np.dtype("float32"): "float",
            np.dtype("float64"): "double"}


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
        lines = textwrap.wrap(initial_indent=self._indent, subsequent_indent=self._indent, break_long_words=False,
                              text=line)
        self.line_buffer.extend(lines)

    def flush(self):
        if self.line_buffer:
            output_gen = "\n".join(line for line in self.line_buffer)

            pathlib.Path(self.path).write_text(output_gen)
            self.line_buffer.clear()


def gen_setup(out_dir: pathlib.Path):
    emitter = Emitter(out_dir.joinpath("setup.py"))
    with emitter.flush_on_exit():
        emitter.print_line("from glob import glob")
        emitter.print_line("from setuptools import setup")
        emitter.print_line("from pybind11.setup_helpers import Pybind11Extension")
        emitter.blank_lines(count=2)
        emitter.print_line("ext_modules = [Pybind11Extension(sorted(glob('*.cpp')))]")
        emitter.print_line("setup(ext_modules=ext_modules)")
        emitter.blank_lines(count=1)


def get_c_array_type(array_type: ir.ArrayType):
    c_dtype = npy_map[array_type.dtype]
    return f"py::array<{c_dtype}>"


@singledispatch
def render(node):
    # some compound stuff has to be broken up if we don't have a replacement
    # (possibly inlinable) sub-routine
    msg = f"No method implemented to render type: {type(node)}."
    raise NotImplementedError(msg)


@render.register
def _(node: ir.AND):
    expr = " && ".join(render(subexpr) for subexpr in node.subexprs)
    return expr


@render.register
def _(node: ir.OR):
    expr = " || ".join(render(subexpr) for subexpr in node.subexprs)
    return expr


@render.register
def _(node: ir.NOT):
    expr = render(node.operand)
    return f"!{expr}"


@render.register
def _(node: ir.StringConst):
    # Todo: need to think about unicode cases
    return f"std::string({node.value})"


@render.register
def _(node: ir.MaxReduction):
    expr = simple_serialize_min_max(node)
    rendered = render(expr)
    return rendered


@render.register
def _(node: ir.MinReduction):
    expr = simple_serialize_min_max(node)
    rendered = render(expr)
    return rendered


@render.register
def _(node: ir.CompareOp):
    left, right = node.subexprs
    left = render(left)
    right = render(right)
    rendered = f"{left} {node.op} {right}"
    return rendered


@render.register
def _(node: ir.BinOp):
    left, right = node.subexprs
    left = render(left)
    right = render(right)
    rendered = f"{left} {node.op} {right}"
    return rendered


@render.register
def _(node: ir.POW):
    left, right = node.subexprs
    left = render(left)
    right = render(right)
    rendered = f"pow({left}, {right})"
    return rendered


@render.register
def _(node: ir.MaxReduction):
    serialized = simple_serialize_min_max(node)
    return render(serialized)


@render.register
def _(node: ir.MinReduction):
    serialized = simple_serialize_min_max(node)
    return render(serialized)


@render.register
def _(node: ir.Min):
    left, right = node.subexprs
    left = render(left)
    right = render(right)
    rendered = f"std::min({left}, {right})"
    return rendered


@render.register
def _(node: ir.SingleDimRef):
    dim = render(node.dim)
    base = render(node.base)
    rendered = f"{base}.shape({dim})"
    return rendered


@render.register
def _(node: ir.NameRef):
    return node.name


@render.register
def _(node: ir.Constant):
    if isinstance(node.value, (bool, np.bool_)):
        # c casing..
        value = "true" if node.value is True else "false"
    else:
        value = str(node.value)
    return value


@render.register
def _(node: ir.Call):
    func_name = extract_name(node)
    # Todo: correct func replacement
    if func_name == "print":
        # quick temporary hack
        func_name = "py::print"
    args = ", ".join(render(arg) for arg in node.args)
    rendered = f"{func_name}({args})"
    return rendered


@render.register
def _(node: ir.Slice):
    params = ", ".join(render(subexpr) for subexpr in node.subexprs)
    rendered = f"py::slice({params})"
    return rendered


@render.register
def _(node: ir.Tuple):
    params = ", ".join(render(subexpr) for subexpr in node.subexprs)
    rendered = f"py::make_tuple({params})"
    return rendered


@render.register
def _(node: ir.Subscript):
    index = render(node.index)
    base = render(node.value)
    rendered = f"{base}[{index}]"
    return rendered


def render_expr(expr: ir.ValueRef,
                type_helper: TypeHelper,
                call_templates: Optional[Dict] = None,
                render_cache: Optional[Dict[ir.Expression, str]] = None):
    if render_cache is None:
        render_cache = {}
    if call_templates is None:
        call_templates = {}
    for site in walk(expr):
        cached = render_cache.get(site)
        if cached is None:
            # first time seen in post order
            if isinstance(site, ir.Call):
                # before rendering, check that we have a corresponding call template
                template_types = tuple(type_helper.check_type(arg) for arg in site.args)
                template = call_templates.get(template_types)
                if template is None:
                    func_name = extract_name(site)
                    msg = f"No call template for call to {func_name} with arguments {template}."
                    raise CompilerError(msg)
                render_cache[site] = render(site)
            else:
                render_cache[site] = render(site)
    return render_cache[expr]


# Todo: pybind11 source shows examples using these types internally. See how this interacts with numpy..

c_type_codes = {
    np.dtype("bool"): "bool",
    np.dtype("int8"): "char",
    np.dtype("uint8"): "uchar",
    np.dtype("int32"): "int32_t",
    np.dtype("int64"): "int64_t",
    np.dtype("float32"): "float",
    np.dtype("float64"): "double",
}

# These are used for array initialization
npy_dtype_codes = {}


def get_c_type_name(type_):
    if isinstance(type_, ir.ArrayType):
        dtype = c_type_codes.get(type_.dtype)
        if dtype is not None:
            return f"py::array_t<{dtype}>"
    else:
        dtype = c_type_codes.get(type_)
        if dtype is not None:
            return dtype
    msg = f"Type {type_} does not have an available C scalar conversion."
    raise CompilerError(msg)


def get_function_header(func: ir.Function, symbols: symbol_table, mangled_name: Optional[str] = None):
    func_name = extract_name(func) if mangled_name is None else mangled_name
    return_type = check_return_type(func, symbols)
    args = []
    for arg in func.args:
        arg_str = extract_name(arg)
        arg_type = symbols.check_type(arg)
        c_type = get_c_type_name(arg_type)
        c_arg_str = f"{c_type} {arg_str}"
        args.append(c_arg_str)
    args_str = ", ".join(a for a in args)
    if return_type is None:
        return_type = "py::object"
    else:
        return_type = npy_map[return_type]
    func_str = f"{return_type} {func_name} ({args_str})"
    return func_str


class FuncWriter:
    # too dangerous to inherit here.. as it'll just leave a blank on unsupported
    def __init__(self, emitter: Emitter, symbols: symbol_table, mangled_name: Optional[str] = None):
        self.emitter = emitter
        self.declared = set()
        self.cached_exprs = {}
        self.mangled_name = mangled_name
        self.symbols = symbols
        self.type_helper = TypeHelper(symbols)

    def __call__(self, node: ir.Function):
        assert isinstance(node, ir.Function)
        self.visit(node)

    def render_expr(self, expr: ir.ValueRef):
        return render_expr(expr, self.type_helper, self.cached_exprs)

    def get_output_func_name(self, node: ir.Function):
        assert isinstance(node, ir.Function)
        return self.mangled_name if self.mangled_name is not None else extract_name(node)

    @singledispatchmethod
    def visit(self, node):
        msg = f"No method to write node of type {type(node)}."
        raise CompilerError(msg)

    @visit.register
    def _(self, node: list):
        for stmt in node:
            self.visit(stmt)

    @visit.register
    def _(self, node: ir.Function):
        func_name = self.get_output_func_name(node)
        header = get_function_header(node, self.symbols, func_name)
        with self.emitter.curly_braces(line=header, semicolon=False):
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.InPlaceOp):
        assert isinstance(node.expr, ir.BinOp)
        if isinstance(node.expr, ir.POW):
            expr = render(node.expr)
            target = render(node.target)
            stmt = f"{target} = {expr};"
        else:
            op = f"{node.expr.op}="
            target = render(node.target)
            value = render(node.value)
            stmt = f"{target} {op} {value}"
        self.emitter.print_line(stmt)

    @visit.register
    def _(self, node: ir.IfElse):
        # render test
        test_str = self.render_expr(node.test)
        header = f"if({test_str})"
        with self.emitter.curly_braces(line=header, semicolon=False):
            self.visit(node.if_branch)
        if node.else_branch:
            with self.emitter.curly_braces(line="else", semicolon=False):
                self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.ForLoop):
        # this should be reduced to a simple range loop
        if not isinstance(node.iterable, ir.AffineSeq) or not isinstance(node.target, ir.NameRef):
            msg = f"Unsupported for loop {node}"
            raise ValueError(msg)
        start, stop, step = node.iterable.subexprs
        start_value_str = self.render_expr(start)
        if node.target in self.declared:
            start_str = f"{node.target.name} = {start_value_str}"
        else:
            target_type = self.symbols.check_type(node.target)
            c_target_type = get_c_type_name(target_type)
            start_str = f"{c_target_type} {node.target.name} = {start_value_str}"
        stop_value_str = self.render_expr(stop)
        stop_str = f"{node.target.name} < {stop_value_str}"
        step_value_str = self.render_expr(step)
        if step == ir.One:
            step_str = f"++{node.target.name}"
        else:
            step_str = f"{node.target.name} += {step_value_str}"
        header = f"for({start_str}; {stop_str}; {step_str})"
        with self.emitter.curly_braces(line=header, semicolon=False):
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.SingleExpr):
        rendered = render(node.expr)
        rendered = f"{rendered};"
        self.emitter.print_line(rendered)

    @visit.register
    def _(self, node: ir.WhileLoop):
        test = self.render_expr(node.test)
        header = f"while({test})"
        with self.emitter.curly_braces(line=header, semicolon=False):
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.Assign):
        # Todo: This needs a render bind and render write for cases where these operations require specialized syntax
        # check if declared
        decl_required = isinstance(node.target, ir.NameRef) and node.target not in self.declared
        # check if types compatible
        target_type = self.type_helper.check_type(node.target)
        value_type = self.type_helper.check_type(node.value)
        cast_required = target_type != value_type
        # allow writing scalar to scalar reference arising from subscripting
        static_cast_allowed = not isinstance(target_type, ir.ArrayType)
        if cast_required and not static_cast_allowed:
            msg = f"Casts from {target_type} to {value_type} are unsupported."
            raise CompilerError(msg)
        value_str = self.render_expr(node.value)
        target_str = self.render_expr(node.target)
        c_target_type = get_c_type_name(target_type)
        if cast_required:
            value_str = f"static_cast<{c_target_type}>({value_str})"
        if decl_required:
            target_str = f"{c_target_type} {target_str}"
        stmt_str = f"{target_str} = {value_str};"
        self.emitter.print_line(stmt_str)

    @visit.register
    def _(self, node: ir.Break):
        self.emitter.print_line("break;")

    @visit.register
    def _(self, node: ir.Return):
        if node.value is None:
            self.emitter.print_line("return py::none();")
        else:
            self.emitter.print_line(f"return {render(node.value)};")


def gen_module_def(emitter: Emitter, module_name: str, func_names: Set[str], docs: Optional[str] = None):
    with emitter.curly_braces(line=f"PYBIND11_MODULE({module_name}, m)"):
        if docs is not None:
            emitter.print_line(line=f"m.doc() = {docs};")
        for func_name in func_names:
            emitter.print_line(f'm.def({func_name}, &{func_name}, "");')


def gen_module(path: pathlib.Path, module_name: str, funcs: List[ir.Function], symbol_tables: Dict[str, symbol_table],
               docs: Optional[str] = None):

    if not module_name.endswith(".cpp"):
        module_name = f"{module_name}.cpp"
    emitter = Emitter(path.joinpath(module_name))
    func_names = {func.name for func in funcs}

    # Todo: don't write if exception occurs
    with emitter.flush_on_exit():
        # boilerplate
        emitter.print_line(line="#include <algorithm>")
        emitter.print_line(line="#include <pybind11/pybind11.h>")
        emitter.print_line(line="#include <pybind11/numpy.h>")
        emitter.blank_lines(count=1)
        emitter.print_line(line="namespace py = pybind11;")
        emitter.blank_lines(count=2)

        for func in funcs:
            symbols = symbol_tables[func.name]
            func_writer = FuncWriter(emitter, symbols, func.name)
            func_writer(func)
            emitter.blank_lines(count=2)

        gen_module_def(emitter, module_name, func_names, docs)
