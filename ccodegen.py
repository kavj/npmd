import pathlib
import textwrap

from contextlib import contextmanager
from functools import singledispatchmethod

import ir

import type_resolution as tr

from errors import CompilerError
from pretty_printing import binop_ordering
from symbol_table import symbol_table
from type_inference import ExprTypeInfer
from visitor import StmtVisitor

"""

most of the transforms are done elsewhere
This should load the appropriate ISA for simd ops and

This still needs a customized visitor to generate statements.
At this point, statements are serialized to the level expected by C code.

"""

std_include_headers = ["Python.h", "numpy/arrayobject.h", "stdint.h", "stdbool.h", "math.h"]


# This assumes that C99 floats are 32 bits and C doubles are 64 bits.
# Numpy seems to depend on this anyway and probably won't compile on
# platforms where this does not hold.


# Todo: need to write codegen for module setup and method declaration in addition to header imports


scalar_type_map = {tr.Int32: "int32_t",
                   tr.Int64: "int64_t",
                   tr.Float32: "float",
                   tr.Float64: "double",
                   tr.Predicate32: "bool",
                   tr.Predicate64: "bool",
                   tr.BoolType: "bool"}


def parenthesized(expr):
    # This is its own thing to avoid too much inline text formatting
    # and allow for a more detailed implementation if redundancy
    # issues ever arise.
    return f"({expr})"


class CodeBuffer:

    def __init__(self, path, indent="    ", max_line_width=70):
        self.ctx = context
        self.path = path
        self.single_indent = indent
        self.max_line_width = max_line_width
        self.line_formatter = textwrap.TextWrapper(tabsize=4, break_long_words=False, break_on_hyphens=False)
        self.line_buffer = []

    @property
    def indent(self):
        return self.line_formatter.initial_indent

    @indent.setter
    def indent(self, indent_):
        self.line_formatter.initial_indent = indent_
        self.line_formatter.subsequent_indent = indent_

    @property
    def indent_len(self):
        return len(self.line_formatter.initial_indent)

    @contextmanager
    def indented(self):
        base_indent = self.indent
        scoped_indent = f"{base_indent}{self.single_indent}"
        self.indent = scoped_indent
        yield
        assert self.indent == scoped_indent
        self.indent = base_indent

    def print_line(self, line):
        lines = self.line_formatter.wrap(line)
        self.line_buffer.extend(lines)

    def write_to(self, path):
        if self.line_buffer:
            output_gen = "\n".join(line for line in self.line_buffer)
            pathlib.Path(path).write_text(output_gen)
            self.line_buffer.clear()


def else_is_elif(stmt: ir.IfElse):
    if len(stmt.else_branch) == 1:
        if isinstance(stmt.else_branch[0], ir.IfElse):
            return True
    return False


def format_header(prefix, cond):
    if cond is None:
        return f"{prefix}{'{'}"
    else:
        return f"{prefix} ({cond}){'{'}"


# Todo: need utilities to check for cases where output type does not match input operand types
#       C99 has totally different type promotion rules, so it's better to break expressions
#       and add indicators to determine exact cast types.

# Todo: we need lowering for overflow checked arithmetic. It might be better to provide most of this
#       via a header.


class pretty_formatter:
    """
    Pretty printer for C99. This does not support tuples or variable length min/max.

    """

    def __call__(self, node):
        expr = self.visit(node)
        return expr

    @singledispatchmethod
    def visit(self, node):
        msg = f"No method to convert node: {node} to C99 code."
        raise NotImplementedError(msg)

    @visit.register
    def _(self, node: ir.Max):
        assert len(node.values) == 2

    @visit.register
    def _(self, node: ir.Min):
        assert len(node.values) == 2

    @visit.register
    def _(self, node: ir.Select):
        test = self.visit(node.test)
        if isinstance(node.test, ir.Select):
            test = parenthesized(test)
        on_true = self.visit(node.on_true)
        if isinstance(node.on_true, (ir.Select, ir.Tuple)):
            on_true = parenthesized(on_true)
        on_false = self.visit(node.on_false)
        if isinstance(node.on_false, (ir.Select, ir.Tuple)):
            on_false = parenthesized(on_false)
        expr = f"{on_true} if {test} else {on_false}"
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
                    left = parenthesized(left)
            elif isinstance(node.left, (ir.BoolOp, ir.CompareOp, ir.Select)):
                left = parenthesized(left)
            if isinstance(node.right, ir.BinOp):
                if op_ordering < binop_ordering[right.op]:
                    left = parenthesized(right)
            elif isinstance(node.right, (ir.BoolOp, ir.CompareOp, ir.Select)):
                right = parenthesized(right)
        expr = f"{left} {op} {right}"
        return expr

    @visit.register
    def _(self, node: ir.CompareOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.left, (ir.BoolOp, ir.CompareOp, ir.Select, ir.Tuple)):
            left = parenthesized(left)
        if isinstance(node.right, (ir.BoolOp, ir.CompareOp, ir.Select, ir.Tuple)):
            right = parenthesized(right)
        expr = f"{left} {node.op} {right}"
        return expr

    @visit.register
    def _(self, node: ir.AND):
        operands = []
        for operand in node.operands:
            formatted = self.visit(operand)
            if isinstance(operand, (ir.AND, ir.OR, ir.Select)):
                formatted = parenthesized(formatted)
            operands.append(formatted)
        expr = " && ".join(operand for operand in operands)
        return expr

    @visit.register
    def _(self, node: ir.OR):
        operands = []
        for operand in node.operands:
            formatted = self.visit(operand)
            if isinstance(operand, ir.Select):
                formatted = parenthesized(formatted)
            operands.append(formatted)
        expr = " || ".join(operand for operand in operands)
        return expr

    @visit.register
    def _(self, node: ir.NOT):
        formatted = self.visit(node.operand)
        if isinstance(node.operand, (ir.AND, ir.OR, ir.Select)):
            formatted = parenthesized(formatted)
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
                operand = parenthesized(operand)
        elif isinstance(node.operand, (ir.UnaryOp, ir.BoolOp, ir.Select)):
            # if we have an unfolded double unary expression such as --,
            # '--expr' would be correct but it's visually jarring. Adding
            # unnecessary parentheses makes it '-(-expr)'.
            operand = parenthesized(operand)
        expr = f"{op}({operand})"
        return expr


class CodeGenBase:

    def __init__(self, ctx, dest):
        self.printer = CodeEmitter(ctx, dest)
        self.format = pretty_formatter()

    @contextmanager
    def indented(self):
        with self.printer.indented():
            yield

    @contextmanager
    def closing_brace(self):
        with self.printer.indented():
            yield
        self.printer.print_line("}")


def mangle_function_name(func, argtypes):
    """
    Produce a C function name based on the argument types, which conforms to
    internal ABI conventions.
    """

    raise NotImplementedError("in my todo list")


def c_decl_from_symbol(sym):
    t = find_c_type(sym.type_)
    return f"{t} {sym.name}"


def gen_arg_decls(symbols: symbol_table):
    return ", ".join(c_decl_from_symbol(s) for s in symbols.arguments)


def gen_variable_decls(symbols):
    return ";\n".join(c_decl_from_symbol(s) for s in symbols.source_locals)


class CModuleCodeGen(CCodeGenBase, StmtVisitor):

    def __init__(self):
        self.format = pretty_formatter()
        self.symbols = None

    @contextmanager
    def function_context(self, symbols):
        assert self.symbols is None
        self.symbols = symbols
        yield
        self.symbols = None

    def __call__(self, func: ir.Function, symbols: symbol_table):
        assert symbols.namespace == func.name
        with self.function_context(symbol_table):
            self.visit(func)

    @contextmanager
    def scoped(self, prefix, cond):
        if cond is None:
            line = f"{prefix}{'{'}"
        else:
            line = f"{prefix} ({cond}){'{'}"
        self.printer.print_line(line)
        with self.printer.closing_brace():
            yield

    @singledispatchmethod
    def visit(self, node):
        raise NotImplementedError

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

        # Todo: need to determine how much of the numpy api should be directly exposed.
        #    At present, I am guessing anything that doesn't require the gil may be exposed,
        #    unless the use of restrict or __restrict__ becomes necessary
        #    (supported on all compilers that compile CPython).
        if isinstance(node.target, ir.NameRef) and node.target not in self.declared:
            if node.in_place:
                msg = f"Inplace assignment cannot be performed against unbound variables, line: {node.pos.line_begin}."
                raise CompilerError(msg)
            # For now, assume C99 back end,
            # compliant with PEP 7
            type_ = self.ctx.get_type(target)
            ctype_ = scalar_type_map.get(type_)
            target = f"{ctype_} {target}"
        stmt = f"{target} = {value};"
        self.printer.print_line(stmt)

    def visit_elif(self, node: ir.IfElse):
        test = self.visit(node.test)
        with self.scoped("else if", test):
            self.visit(node.if_branch)
        if else_is_elif(node):
            self.visit_elif(node.else_branch[0])
        elif node.else_branch:
            with self.scoped("else", None):
                self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.IfElse):
        test = self.visit(node.test)
        with self.scoped("if", test):
            self.visit(node.if_branch)
        if else_is_elif(node):
            self.visit_elif(node)
        elif node.else_branch:
            with self.scoped("else", None):
                self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.ForLoop):
        # Printing is more restrictive, check that
        # we have a supported loop structure
        assert isinstance(node.target, ir.NameRef)
        assert isinstance(node.iterable, (ir.AffineSeq, ir.Reversed))
        # check for unit step
        target = self.format_lvalue_ref(node.target)
        # check whether we can use ++ op
        # insufficient support for reversed() thus far.
        if node.iterable.step == ir.One:
            step_expr = f"++{target}"
        else:
            increm_by = self.format(node.iterable.step)
            step_expr = f"{target} += {increm_by}"
        start = self.format(node.iterable.start)
        stop = self.format(node.iterable.stop)
        # implements range semantics with forward step, with the caveat
        # that any escaping value of target must be copied out in the loop body
        cond = f"{target} = {start}; {target} < {stop}; {step_expr}"
        with self.scoped("for", cond):
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.WhileLoop):
        cond = self.format(node.test)
        with self.scoped("while", cond):
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
        value = self.format(node.value)
        if node.in_place:
            assert isinstance(node.value, ir.BinOp)
            op = node.value.op
        else:
            op = "="
        line = f"{target} {op} {value};"
        self.printer.print_line(line)

    @visit.register
    def _(self, node: ir.SingleExpr):
        expr = self.format(node.expr)
        line = f"{expr};"
        self.printer.print_line(line)


# need a header file generator..

class BoilerplateWriter(CCodeGenBase):

    # This is meant to be controlled by a codegen driver,
    # which manages opening/closing of a real or virtual destination file.

    def __init__(self, ctx, dest):
        super().__init__(ctx, dest)

    def print_sys_header_text(name):
        s = f"#include<{name}>"
        self.print_line(s)

    def print_user_header_text(name):
        s = f"#include \"{name}\""
        self.print_line(s)

    def gen_source_top(sys_headers=(), user_headers=()):
        self.print_line("#define PY_SSIZE_T_CLEAN")
        self.print_sys_header_text("Python.h")
        for h in sys_headers:
            self.print_sys_header_text(h)
        for h in user_headers:
            self.print_user_header_text(h)

    def gen_module_init(modname):
        if modname == "mod":
            raise CompilerError("mod is treated as a reserved name.")
        self.print_line(f"PyMODINIT_FUNC PyInit_{modname}(void){'{'}")
        with self.closing_brace():
            self.print_line(f"PyObject* mod = PyModule_Create(&{modname});")
            printline("if(mod == NULL){")
            with self.closing_brace():
                printline("return NULL;")

    def gen_method_table(modname, funcs):
        # no keyword support..
        self.print_line(f"static PyMethodDef {modname}Methods[] = {'{'}")
        with self.indented():
            for name, doc in funcs:
                if doc is None:
                    self.print_line(f"{name}, {modname}_{name}, METH_VARARGS, NULL{'{'}")
                else:
                    self.print_line(f"{name}, {modname}_{name}, METH_VARARGS, {doc}{'{'}")
        self.print_line("};")
        # sentinel ending entry
        self.print_line("{NULL, NULL, 0, NULL}")

    def gen_module_def(modname):
        self.print_line(f"static PyModuleDef {modname} = {'{'}")
        with self.indented():
            self.print_line("PyModuleDef_HEAD_INIT,")
            self.print_line(f"{modname},")
            self.print_line("NULL,")
            self.print_line("-1")
            self.print_line(f"{modname}Methods")
