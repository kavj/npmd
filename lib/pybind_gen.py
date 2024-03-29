import pathlib
import textwrap

import numpy

from contextlib import contextmanager
from functools import singledispatchmethod
from typing import Dict, List, Optional, Union

import lib.ir as ir

from lib.canonicalize import serialize_min_max
from lib.errors import CompilerError
from lib.symbol_table import SymbolTable
from lib.type_checks import TypeHelper, check_return_type


npy_c_type_codes = {
    numpy.dtype('bool'): 'NPY_BOOL',
    numpy.dtype('int8'): 'NPY_INT8',
    numpy.dtype('uint8'): 'NPY_UINT8',
    numpy.dtype('int32'): 'NPY_INT32',
    numpy.dtype('int64'): 'NPY_INT64',
    numpy.dtype('float32'): 'NPY_FLOAT32',
    numpy.dtype('float64'): 'NPY_FLOAT64',
}


npy_map = {numpy.dtype('bool'): 'bool',
           numpy.dtype('int8'): 'int8_t',
           numpy.dtype('uint8'): 'uint8_t',
           numpy.dtype('int32'): 'int32_t',
           numpy.dtype('int64'): 'int64_t',
           numpy.dtype('float32'): 'float',
           numpy.dtype('float64'): 'double'}


compare_ops = {ir.GT: '>',
               ir.GE: '>=',
               ir.LT: '<',
               ir.LE: '<=',
               ir.EQ: '==',
               ir.NE: '!=',
               ir.IN: 'in',
               ir.NOTIN: 'not in'}


arithmetic_ops = {ir.ADD: '+',
                  ir.SUB: '-',
                  ir.MULT: '*',
                  ir.TRUEDIV: '/',
                  ir.FLOORDIV: '//',
                  ir.MOD: '%',
                  ir.POW: '**',
                  ir.LSHIFT: '<<',
                  ir.RSHIFT: '>>',
                  ir.BITOR: '|',
                  ir.BITXOR: '^',
                  ir.BITAND: '&',
                  ir.MATMULT: '@'
                  }


class Emitter:

    def __init__(self, path, indent='    ', max_line_width=70):
        self._indent = ''
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
            self.line_buffer.append('')

    @contextmanager
    def ifdef_directive(self, cond):
        line = f'#ifdef {cond}'
        self.print_line(line)
        yield
        self.print_line('#endif')

    @contextmanager
    def ifndef_directive(self, cond):
        line = f'#ifndef {cond}'
        self.print_line(line)
        yield
        self.print_line('#endif')

    @contextmanager
    def indented(self):
        self._indent = f'{self._indent}{self.single_indent}'
        yield
        self._indent = self._indent[:-len(self.single_indent)]

    @contextmanager
    def scope(self, line=None, semicolon=False):
        if line is not None:
            line = f'{line} {"{"}'
        else:
            line = '{'
        self.print_line(line)
        with self.indented():
            with self.decls.scope():
                yield
        if semicolon:
            self.print_line('};')
        else:
            self.print_line('}')

    @contextmanager
    def flush_on_exit(self):
        yield
        self.flush()

    def declare(self, name: ir.NameRef):
        self.decls.declare(name)

    def is_declared(self, name: ir.NameRef):
        return self.decls.is_declared(name)

    def print_line(self, line, terminate=False):
        if terminate:
            line += ';'
        lines = textwrap.wrap(initial_indent=self._indent, subsequent_indent=self._indent, break_long_words=False, text=line)
        self.line_buffer.extend(lines)

    def flush(self):
        if self.line_buffer:
            output_gen = '\n'.join(line for line in self.line_buffer)

            pathlib.Path(self.path).write_text(output_gen)
            self.line_buffer.clear()


def gen_setup(out_dir: pathlib.Path, modname):
    emitter = Emitter(out_dir.joinpath("setup.py"))
    with emitter.flush_on_exit():
        emitter.print_line('from glob import glob')
        emitter.print_line('from setuptools import setup')
        emitter.print_line('from pybind11.setup_helpers import Pybind11Extension')
        emitter.blank_lines(count=2)
        ext = f'ext_modules = [Pybind11Extension("{modname}", sorted(glob("*.cpp")))]'
        emitter.print_line(ext)
        emitter.print_line('setup(ext_modules=ext_modules)')
        emitter.blank_lines(count=1)


def get_c_array_type(array_type: ir.ArrayType):
    c_dtype = npy_map[array_type.dtype]
    return f'py::array_t<{c_dtype}, py::array::c_style | py::array::forcecast>'


# Todo: pybind11 source shows examples using these types internally. See how this interacts with numpy..

c_type_codes = {
    numpy.dtype('bool'): 'bool',
    numpy.dtype('int8'): 'char',
    numpy.dtype('uint8'): 'uint8_t',
    numpy.dtype('int32'): 'int32_t',
    numpy.dtype('int64'): 'int64_t',
    numpy.dtype('float32'): 'float',
    numpy.dtype('float64'): 'double',
}


def format_type(node: Union[numpy.dtype, ir.ArrayType]):
    if isinstance(node, numpy.dtype):
        return c_type_codes[node]
    else:
        dtype = c_type_codes[node.dtype]
        return f'py::array_t<{dtype}, py::array::c_style | py::array::forcecast>'


def get_c_type_name(type_):
    if isinstance(type_, ir.ArrayType):
        dtype = c_type_codes.get(type_.dtype)
        if dtype is not None:
            return f'py::array_t<{dtype}, py::array::c_style | py::array::forcecast>'
    else:
        dtype = c_type_codes.get(type_)
        if dtype is not None:
            return dtype
    msg = f'Type {type_} does not have an available C scalar conversion.'
    raise CompilerError(msg)


def get_function_header(func: ir.Function, symbols: SymbolTable, mangled_name: Optional[str] = None):
    func_name = func.name if mangled_name is None else mangled_name
    return_type = check_return_type(func)
    args = []
    for arg in func.args:
        arg_str = arg.name
        arg_type = symbols.check_type(arg)
        c_type = get_c_type_name(arg_type)
        c_arg_str = f'{c_type} {arg_str}'
        args.append(c_arg_str)
    args_str = ', '.join(a for a in args)
    if isinstance(return_type, ir.NoneRef):
        return_type = 'py::object'
    else:
        return_type = format_type(return_type)
    func_str = f'{return_type} {func_name} ({args_str})'
    return func_str


class ExprFormatter:
    def __init__(self, symbols: SymbolTable):
        self.type_helper = TypeHelper(symbols)

    @singledispatchmethod
    def render(self, node):
        # some compound stuff has to be broken up if we don't have a replacement
        # (possibly inlinable) sub-routine
        msg = f'No method implemented to render type: {type(node)}.'
        raise NotImplementedError(msg)

    @render.register
    def _(self, node: ir.NoneRef):
        return 'py::none()'

    @render.register
    def _(self, node: ir.ArrayAlloc):
        dtype = c_type_codes[numpy.dtype(node.dtype.value)]
        if isinstance(node.shape, ir.TUPLE):
            if len(node.shape) == 1:
                dim, = node.shape.subexprs
                shape = self.render(dim)
            else:
                dims = ', '.join(self.render(d) for d in node.shape.subexprs)
                shape = f'py::array::ShapeContainer {{{dims}}}'
        else:
            shape = self.render(node.shape)
        return f'py::array_t<{dtype}>({shape})'

    @render.register
    def _(self, node: ir.CAST):
        src_type = self.type_helper(node.value)
        target_type = node.target_type
        static_cast_allowed = not isinstance(target_type, ir.ArrayType)
        if not static_cast_allowed:
            msg = f'Casts from {target_type} to {src_type} are unsupported.'
            raise CompilerError(msg)
        expr_str = self.render(node.value)
        c_target_type = get_c_type_name(target_type)
        expr_str = f'static_cast<{c_target_type}>({expr_str})'
        return expr_str

    @render.register
    def _(self, node: ir.CONSTANT):
        return str(node.value)

    @render.register
    def _(self, node: ir.AND):
        expr = ' && '.join(self.render(subexpr) for subexpr in node.subexprs)
        return expr

    @render.register
    def _(self, node: ir.OR):
        expr = ' || '.join(self.render(subexpr) for subexpr in node.subexprs)
        return expr

    @render.register
    def _(self, node: ir.NOT):
        expr = self.render(node.operand)
        return f'!{expr}'

    @render.register
    def _(self, node: ir.MaxOf):
        expr = serialize_min_max(node)
        rendered = self.render(expr)
        return rendered

    @render.register
    def _(self, node: ir.MinOf):
        expr = serialize_min_max(node)
        rendered = self.render(expr)
        return rendered

    @render.register
    def _(self, node: ir.CompareOp):
        left, right = node.subexprs
        left = self.render(left)
        right = self.render(right)
        op = compare_ops[type(node)]
        rendered = f'{left} {op} {right}'
        return rendered

    @render.register
    def _(self, node: ir.TRUEDIV):
        left, right = node.subexprs
        # We don't have this operator in C++, so we insert cast
        # nodes at the last second as necessary.
        left_type = self.type_helper(left)
        right_type = self.type_helper(right)
        result_type = self.type_helper(node)
        if left_type != result_type:
            left = ir.CAST(left, result_type)
        if right_type != result_type:
            right = ir.CAST(right, result_type)
        left = self.render(left)
        right = self.render(right)
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
        op = arithmetic_ops[type(node)]
        rendered = f'{left} {op} {right}'
        return rendered

    @render.register
    def _(self, node: ir.POW):
        left, right = node.subexprs
        left = self.render(left)
        right = self.render(right)
        rendered = f'pow({left}, {right})'
        return rendered

    @render.register
    def _(self, node: ir.MaxOf):
        serialized = serialize_min_max(node)
        return self.render(serialized)

    @render.register
    def _(self, node: ir.MinOf):
        serialized = serialize_min_max(node)
        return self.render(serialized)

    @render.register
    def _(self, node: ir.MIN):
        left, right = node.subexprs
        left = self.render(left)
        right = self.render(right)
        rendered = f'std::min({left}, {right})'
        return rendered

    @render.register
    def _(self, node: ir.SingleDimRef):
        dim = self.render(node.dim)
        base = self.render(node.base)
        rendered = f'{base}.shape({dim})'
        return rendered

    @render.register
    def _(self, node: ir.NameRef):
        return node.name

    @render.register
    def _(self, node: ir.CONSTANT):
        if isinstance(node.value, (bool, numpy.bool_)):
            # c casing..
            value = 'true' if node.value is True else 'false'
        else:
            value = str(node.value)
        return value

    @render.register
    def _(self, node: ir.Call):
        func_name = node.func
        # Todo: correct func replacement
        if func_name == 'print':
            # quick temporary hack
            func_name = 'py::print'
        args = ', '.join(self.render(arg) for arg in node.args.subexprs)
        rendered = f'{func_name}({args})'
        return rendered

    @render.register
    def _(self, node: ir.Slice):
        params = ', '.join(self.render(subexpr) for subexpr in node.subexprs)
        rendered = f'py::slice({params})'
        return rendered

    @render.register
    def _(self, node: ir.StringConst):
        return f'R"({node.value})"'

    @render.register
    def _(self, node: ir.TUPLE):
        params = ', '.join(self.render(subexpr) for subexpr in node.subexprs)
        rendered = f'py::make_tuple({params})'
        return rendered

    @render.register
    def _(self, node: ir.Subscript):
        index = self.render(node.index)
        base = self.render(node.value)
        subscr_type = self.type_helper(node)
        # this should be good enough for now. Need to actually make
        # loop local pointer access somewhere for cheaper subscripting
        if isinstance(subscr_type, ir.ArrayType):
            rendered = f'{base}[{index}]'
        else:
            rendered = f'{base}.mutable_at({index})'
        return rendered


class FuncWriter:
    def __init__(self, emitter: Emitter, symbols: SymbolTable, mangled_name: Optional[str] = None):
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
            name_str = name.name
            if self.emitter.is_declared(name):
                return name_str
            else:
                # added symbol, must declare in place
                target_type = self.symbols.check_type(name_str)
                c_target_type = get_c_type_name(target_type)
                self.emitter.declare(name)
                return f'{c_target_type} {name_str}'
        else:
            return self.render(name)

    def format_cast(self, expr: ir.ValueRef, src_type, target_type):
        cast_required = src_type != target_type
        # allow writing scalar to scalar reference arising from subscripting
        static_cast_allowed = not isinstance(target_type, ir.ArrayType)
        if cast_required and not static_cast_allowed:
            msg = f'Casts from {target_type} to {src_type} are unsupported.'
            raise CompilerError(msg)
        expr_str = self.render(expr)
        if cast_required:
            c_target_type = get_c_type_name(target_type)
            expr_str = f'static_cast<{c_target_type}>({expr_str})'
        return expr_str

    def render(self, expr):
        return self.formatter.render(expr)

    def get_output_func_name(self, node: ir.Function):
        assert isinstance(node, ir.Function)
        return self.mangled_name if self.mangled_name is not None else node.name

    @singledispatchmethod
    def visit(self, node):
        msg = f'No method to write node of type {type(node)}.'
        raise CompilerError(msg)

    @visit.register
    def _(self, node: list):
        for stmt in node:
            self.visit(stmt)

    @visit.register
    def _(self, node: ir.Function):
        func_name = self.get_output_func_name(node)
        header = get_function_header(node, self.symbols, func_name)
        with self.emitter.scope(line=header, semicolon=False):
            # decl everything local, group by type
            for arg in node.args:
                self.emitter.declare(arg)
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.InPlaceOp):
        # no declaration
        assert isinstance(node.value, ir.BinOp)
        if isinstance(node.value, ir.POW):
            raise CompilerError('currently broken path..')
        else:
            op = arithmetic_ops[type(node.value)]
            op = f'{op}='
            target, value = node.value.subexprs
            # binop expressions are sorted for consistent hashing,
            # but here we need to check the actual target
            if target is not node.target:
                # swap if left and right are not the same expression
                # and they are misordered
                assert value is node.target
                target, value = value, target
            target = self.render(target)
            value = self.render(value)
            stmt = f'{target} {op} {value};'
        self.emitter.print_line(stmt)

    @visit.register
    def _(self, node: ir.IfElse):
        # render test
        test_str = self.render(node.test)
        header = f'if({test_str})'
        # Check for things assigned in both branches yet unbound here
        with self.emitter.decls.scope():
            # check_all_declared(node.if_branch, self.emitter.decls)
            if_decls = self.emitter.decls.innermost()
        with self.emitter.decls.scope():
            # check_all_declared(node.else_branch, self.emitter.decls)
            else_decls = self.emitter.decls.innermost()
        hoist_decls = if_decls.intersection(else_decls)
        for decl in hoist_decls:
            decl_str = self.format_target(decl)
            decl_str_term = f'{decl_str};'
            self.emitter.print_line(decl_str_term)
        with self.emitter.scope(line=header, semicolon=False):
            self.visit(node.if_branch)
        if node.else_branch:
            # Todo: enable flattening to else if
            with self.emitter.scope(line='else', semicolon=False):
                self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.ForLoop):
        # this should be reduced to a simple range loop
        if not isinstance(node.iterable, ir.AffineSeq) or not isinstance(node.target, ir.NameRef):
            msg = f'Unsupported for loop {node}'
            raise ValueError(msg)
        with self.emitter.decls.scope():
            start, stop, step = node.iterable.subexprs
            start_value_str = self.render(start)
            target_decl = self.format_target(node.target)
            # first is in case of declaration here
            start_str = f'{target_decl} = {start_value_str}'
            stop_value_str = self.render(stop)
            stop_str = f'{node.target.name} < {stop_value_str}'
            step_value_str = self.render(step)
            if step == ir.One:
                step_str = f'++{node.target.name}'
            else:
                step_str = f'{node.target.name} += {step_value_str}'
            header = f'for({start_str}; {stop_str}; {step_str})'
            with self.emitter.scope(line=header, semicolon=False):
                self.visit(node.body)

    @visit.register
    def _(self, node: ir.SingleExpr):
        rendered = self.render(node.value)
        self.emitter.print_line(rendered, terminate=True)

    @visit.register
    def _(self, node: ir.WhileLoop):
        test = self.render(node.test)
        header = f'while({test})'
        with self.emitter.scope(line=header, semicolon=False):
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.Assign):
        # Todo: This needs a render bind and render write for cases where these operations require specialized syntax
        # check if declared
        # check if types compatible
        target_str = self.format_target(node.target)
        target_type = self.type_helper(node.target)
        if isinstance(node.target, ir.NameRef):
            if not self.emitter.is_declared(node.target):
                self.emitter.declare(node.target)
                target_str = f'{target_type} {target_str}'
        value_type = self.type_helper(node.value)
        value_str = self.format_cast(node.value, value_type, target_type)
        stmt = f'{target_str} = {value_str};'
        self.emitter.print_line(stmt)

    @visit.register
    def _(self, node: ir.ArrayFill):
        name = self.render(node.array)
        # get size expression for std::fill
        # this is only to be used with contiguous arrays
        sz = f'{name}.size()'
        # we don't have a pointer type in IR yet, thus this
        t = self.type_helper(node.array)
        dt = get_c_type_name(t.dtype)
        buffer = f'static_cast<{dt}*>({name}.mutable_data())'
        self.emitter.print_line(f'std::fill({buffer}, {buffer} + {sz}, {node.fill_value.value});')

    @visit.register
    def _(self, node: ir.Break):
        self.emitter.print_line('break;')

    @visit.register
    def _(self, node: ir.Continue):
        self.emitter.print_line('continue;')

    @visit.register
    def _(self, node: ir.Return):
        self.emitter.print_line(f'return {self.render(node.value)};')


def gen_module_def(emitter: Emitter, module_name: str, funcs: List[ir.Function], docs: Optional[str] = None):
    module_name_no_ext = pathlib.Path(module_name).stem
    with emitter.scope(line=f'PYBIND11_MODULE({module_name_no_ext}, m)'):
        if docs is not None:
            emitter.print_line(line=f'm.doc() = R"({docs})";')
        for func in funcs:
            name = f'm.def(R"({func.name})", &{func.name}'
            header_params = [name]
            if func.docstring is not None:
                docstring = f'R"({func.docstring})"'
                header_params.append(docstring)
            if func.args:
                header_params.extend(f'py::arg(R"({arg.name})")' for arg in func.args)
            header = ', '.join(header_params)
            # add closing notation
            header = f'{header});'
            emitter.print_line(header)


def gen_module(path: pathlib.Path, module_name: str, funcs: List[ir.Function], symbol_tables: Dict[str, SymbolTable],
               docs: Optional[str] = None):
    if not module_name.endswith(".cpp"):
        module_name += '.cpp'
    emitter = Emitter(path.joinpath(module_name))

    # Todo: don't write if exception occurs
    with emitter.flush_on_exit():
        # boilerplate
        emitter.print_line(line='#include <pybind11/pybind11.h>')
        emitter.print_line(line='#include <pybind11/numpy.h>')
        emitter.print_line(line='#include <cstdint>')
        emitter.print_line(line='#include <algorithm>')
        emitter.blank_lines(count=1)
        emitter.print_line(line='namespace py = pybind11;')
        emitter.blank_lines(count=2)

        for func in funcs:
            symbols = symbol_tables[func.name]
            func_writer = FuncWriter(emitter, symbols, func.name)
            func_writer(func)
            emitter.blank_lines(count=2)

        gen_module_def(emitter, module_name, funcs, docs)
