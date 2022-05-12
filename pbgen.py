import pathlib
import textwrap

import numpy as np

from collections import defaultdict
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

    def print_line(self, line, terminate=False):
        if terminate:
            line += ";"
        lines = textwrap.wrap(initial_indent=self._indent, subsequent_indent=self._indent, break_long_words=False,
                              text=line)
        self.line_buffer.extend(lines)

    def flush(self):
        if self.line_buffer:
            output_gen = "\n".join(line for line in self.line_buffer)

            pathlib.Path(self.path).write_text(output_gen)
            self.line_buffer.clear()


def gen_setup(out_dir: pathlib.Path, modname):
    emitter = Emitter(out_dir.joinpath("setup.py"))
    with emitter.flush_on_exit():
        emitter.print_line("from glob import glob")
        emitter.print_line("from setuptools import setup")
        emitter.print_line("from pybind11.setup_helpers import Pybind11Extension")
        emitter.blank_lines(count=2)
        ext = f'ext_modules = [Pybind11Extension("{modname}", sorted(glob("*.cpp")))]'
        emitter.print_line(ext)
        emitter.print_line("setup(ext_modules=ext_modules)")
        emitter.blank_lines(count=1)


def get_c_array_type(array_type: ir.ArrayType):
    c_dtype = npy_map[array_type.dtype]
    return f"py::array<{c_dtype}, py::array::c_style | py::array::forcecast>"


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


def get_c_type_name(type_):
    if isinstance(type_, ir.ArrayType):
        dtype = c_type_codes.get(type_.dtype)
        if dtype is not None:
            return f"py::array_t<{dtype}, py::array::c_style | py::array::forcecast>"
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


class ExprFormatter:
    def __init__(self, symbols: symbol_table):
        self.type_helper = TypeHelper(symbols)

    @singledispatchmethod
    def render(self, node):
        # some compound stuff has to be broken up if we don't have a replacement
        # (possibly inlinable) sub-routine
        msg = f"No method implemented to render type: {type(node)}."
        raise NotImplementedError(msg)

    @render.register
    def _(self, node: ir.Constant):
        return str(node.value)

    @render.register
    def _(self, node: ir.AND):
        expr = " && ".join(self.render(subexpr) for subexpr in node.subexprs)
        return expr

    @render.register
    def _(self, node: ir.OR):
        expr = " || ".join(self.render(subexpr) for subexpr in node.subexprs)
        return expr

    @render.register
    def _(self, node: ir.NOT):
        expr = self.render(node.operand)
        return f"!{expr}"

    @render.register
    def _(self, node: ir.StringConst):
        # Todo: need to think about unicode cases
        return f"std::string({node.value})"

    @render.register
    def _(self, node: ir.MaxReduction):
        expr = simple_serialize_min_max(node)
        rendered = self.render(expr)
        return rendered

    @render.register
    def _(self, node: ir.MinReduction):
        expr = simple_serialize_min_max(node)
        rendered = self.render(expr)
        return rendered

    @render.register
    def _(self, node: ir.CompareOp):
        left, right = node.subexprs
        left = self.render(left)
        right = self.render(right)
        rendered = f"{left} {node.op} {right}"
        return rendered

    @render.register
    def _(self, node: ir.TRUEDIV):
        left, right = node.subexprs
        # need casts here for consistency
        left_type = self.type_helper.check_type(left)
        f64 = np.dtype('float64')
        left = self.render(left)
        if left_type == f64:
            left = f'static_cast<double>({left})'
        right_type = self.type_helper.check_type(right)
        right = self.render(right)
        if right_type == f64:
            # Todo: an earlier pass should instrument cast nodes..
            right = f'static_cast<double>({right})'
        rendered = f'{left} / {right}'
        return rendered

    @render.register
    def _(self, node: ir.FLOORDIV):
        left, right = node.subexprs
        left = self.render(left)
        right = self.render(right)
        rendered = f'{left} / {right}'
        return rendered

    @render.register
    def _(self, node: ir.BinOp):
        left, right = node.subexprs
        left = self.render(left)
        right = self.render(right)
        rendered = f"{left} {node.op} {right}"
        return rendered

    @render.register
    def _(self, node: ir.POW):
        left, right = node.subexprs
        left = self.render(left)
        right = self.render(right)
        rendered = f"pow({left}, {right})"
        return rendered

    @render.register
    def _(self, node: ir.MaxReduction):
        serialized = simple_serialize_min_max(node)
        return self.render(serialized)

    @render.register
    def _(self, node: ir.MinReduction):
        serialized = simple_serialize_min_max(node)
        return self.render(serialized)

    @render.register
    def _(self, node: ir.Min):
        left, right = node.subexprs
        left = self.render(left)
        right = self.render(right)
        rendered = f"std::min({left}, {right})"
        return rendered

    @render.register
    def _(self, node: ir.SingleDimRef):
        dim = self.render(node.dim)
        base = self.render(node.base)
        rendered = f"{base}.shape({dim})"
        return rendered

    @render.register
    def _(self, node: ir.NameRef):
        return node.name

    @render.register
    def _(self, node: ir.Constant):
        if isinstance(node.value, (bool, np.bool_)):
            # c casing..
            value = 'true' if node.value is True else 'false'
        else:
            value = str(node.value)
        return value

    @render.register
    def _(self, node: ir.StringConst):
        return f'"{node.value}"'

    @render.register
    def _(self, node: ir.Call):
        func_name = extract_name(node)
        # Todo: correct func replacement
        if func_name == 'print':
            # quick temporary hack
            func_name = 'py::print'
        args = ", ".join(self.render(arg) for arg in node.args)
        rendered = f'{func_name}({args})'
        return rendered

    @render.register
    def _(self, node: ir.Slice):
        params = ", ".join(self.render(subexpr) for subexpr in node.subexprs)
        rendered = f"py::slice({params})"
        return rendered

    @render.register
    def _(self, node: ir.Tuple):
        params = ", ".join(self.render(subexpr) for subexpr in node.subexprs)
        rendered = f"py::make_tuple({params})"
        return rendered

    @render.register
    def _(self, node: ir.Subscript):
        index = self.render(node.index)
        base = self.render(node.value)
        rendered = f"{base}[{index}]"
        return rendered


class FuncWriter:
    # too dangerous to inherit here.. as it'll just leave a blank on unsupported
    def __init__(self, emitter: Emitter, symbols: symbol_table, mangled_name: Optional[str] = None):
        self.emitter = emitter
        self.cached_exprs = {}
        self.mangled_name = mangled_name
        self.symbols = symbols
        self.type_helper = TypeHelper(symbols)
        self.formatter = ExprFormatter(self.symbols)

    def __call__(self, node: ir.Function):
        assert isinstance(node, ir.Function)
        self.visit(node)

    def format_target(self, name: ir.ValueRef):
        if isinstance(name, ir.NameRef):
            name_str = extract_name(name)
            if self.symbols.is_source_name(name):
                # these are declared at the start of a scope
                return name_str
            else:
                # added symbol, must declare in place
                target_type = self.symbols.check_type(name_str)
                c_target_type = get_c_type_name(target_type)
                return f"{c_target_type} {name_str}"
        else:
            return self.render(name)

    def format_cast(self, expr: ir.ValueRef, src_type, target_type):
        cast_required = src_type != target_type
        # allow writing scalar to scalar reference arising from subscripting
        static_cast_allowed = not isinstance(target_type, ir.ArrayType)
        if cast_required and not static_cast_allowed:
            msg = f"Casts from {target_type} to {src_type} are unsupported."
            raise CompilerError(msg)
        expr_str = self.render(expr)
        if cast_required:
            c_target_type = get_c_type_name(target_type)
            expr_str = f"static_cast<{c_target_type}>({expr_str})"
        return expr_str

    def render(self, expr):
        return self.formatter.render(expr)

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
            # decl everything local, group by type
            decls = defaultdict(list)
            for sym in self.symbols.source_locals:
                target_type = self.symbols.check_type(sym.name)
                decls[target_type].append(sym.name)
            for target_type, names in decls.items():
                c_target_type = get_c_type_name(target_type)
                name_decls = ", ".join(names)
                decl_str = f"{c_target_type} {name_decls};"
                self.emitter.print_line(decl_str)
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.InPlaceOp):
        # no declaration
        assert isinstance(node.expr, ir.BinOp)
        if isinstance(node.expr, ir.POW):
            expr = self.render(node.expr)
            target = self.render(node.target)
            stmt = f"{target} = {expr};"
        else:
            op = f"{node.expr.op}="
            target = self.render(node.target)
            value = self.render(node.value)
            stmt = f"{target} {op} {value};"
        self.emitter.print_line(stmt)

    @visit.register
    def _(self, node: ir.IfElse):
        # render test
        test_str = self.render(node.test)
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
        start_value_str = self.render(start)
        target = self.format_target(node.target)
        start_str = f"{target} = {start_value_str}"
        stop_value_str = self.render(stop)
        stop_str = f"{node.target.name} < {stop_value_str}"
        step_value_str = self.render(step)
        if step == ir.One:
            step_str = f"++{node.target.name}"
        else:
            step_str = f"{node.target.name} += {step_value_str}"
        header = f"for({start_str}; {stop_str}; {step_str})"
        with self.emitter.curly_braces(line=header, semicolon=False):
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.SingleExpr):
        rendered = self.render(node.expr)
        self.emitter.print_line(rendered, terminate=True)

    @visit.register
    def _(self, node: ir.WhileLoop):
        test = self.render(node.test)
        header = f"while({test})"
        with self.emitter.curly_braces(line=header, semicolon=False):
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.Assign):
        # Todo: This needs a render bind and render write for cases where these operations require specialized syntax
        # check if declared
        # check if types compatible
        target_str = self.format_target(node.target)
        target_type = self.type_helper.check_type(node.target)
        value_type = self.type_helper.check_type(node.value)
        value_str = self.format_cast(node.value, value_type, target_type)
        stmt = f"{target_str} = {value_str};"
        return stmt

    @visit.register
    def _(self, node: ir.Break):
        self.emitter.print_line("break;")

    @visit.register
    def _(self, node: ir.Return):
        if node.value is None:
            self.emitter.print_line("return py::none();")
        else:
            self.emitter.print_line(f"return {self.render(node.value)};")


def gen_module_def(emitter: Emitter, module_name: str, func_names: Set[str], docs: Optional[str] = None):
    module_name_no_ext = pathlib.Path(module_name).stem
    with emitter.curly_braces(line=f"PYBIND11_MODULE({module_name_no_ext}, m)"):
        if docs is not None:
            emitter.print_line(line=f"m.doc() = {docs};")
        for func_name in func_names:
            emitter.print_line(f'm.def("{func_name}", &{func_name}, "");')


def gen_module(path: pathlib.Path, module_name: str, funcs: List[ir.Function], symbol_tables: Dict[str, symbol_table],
               docs: Optional[str] = None):

    if not module_name.endswith(".cpp"):
        module_name += ".cpp"
    emitter = Emitter(path.joinpath(module_name))
    func_names = {func.name for func in funcs}

    # Todo: don't write if exception occurs
    with emitter.flush_on_exit():
        # boilerplate
        emitter.print_line(line="#include <pybind11/pybind11.h>")
        emitter.print_line(line="#include <pybind11/numpy.h>")
        emitter.print_line(line="#include <algorithm>")
        emitter.blank_lines(count=1)
        emitter.print_line(line="namespace py = pybind11;")
        emitter.blank_lines(count=2)

        for func in funcs:
            symbols = symbol_tables[func.name]
            func_writer = FuncWriter(emitter, symbols, func.name)
            func_writer(func)
            emitter.blank_lines(count=2)

        gen_module_def(emitter, module_name, func_names, docs)
