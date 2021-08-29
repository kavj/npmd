import ast
import sys
import typing
import symtable
import warnings

from contextlib import contextmanager
from itertools import islice
from pathlib import Path

import ir
from canonicalize import replace_call
from errors import CompilerError
from lowering import const_folding
from symbol_table import st_from_pyst
from utils import unpack_assignment, unpack_iterated, wrap_input


binary_ops = {ast.Add: "+",
              ast.Sub: "-",
              ast.Mult: "*",
              ast.Div: "/",
              ast.FloorDiv: "//",
              ast.Mod: "%",
              ast.Pow: "**",
              ast.LShift: "<<",
              ast.RShift: ">>",
              ast.BitOr: "|",
              ast.BitXor: "^",
              ast.BitAnd: "&",
              ast.MatMult: "@"}

binary_in_place_ops = {ast.Add: "+=",
                       ast.Sub: "-=",
                       ast.Mult: "*=",
                       ast.Div: "/=",
                       ast.FloorDiv: "//=",
                       ast.Mod: "%=",
                       ast.Pow: "**=",
                       ast.LShift: "<<=",
                       ast.RShift: ">>=",
                       ast.BitOr: "|=",
                       ast.BitXor: "^=",
                       ast.BitAnd: "&=",
                       ast.MatMult: "@="}

unary_ops = {ast.UAdd: "+",
             ast.USub: "-",
             ast.Invert: "~",
             ast.Not: "not"}

bool_ops = {ast.And: "and",
            ast.Or: "or"}

compare_ops = {ast.Eq: "==",
               ast.NotEq: "!=",
               ast.Lt: "<",
               ast.LtE: "<=",
               ast.Gt: ">",
               ast.GtE: ">=",
               ast.Is: "is",
               ast.IsNot: "isnot",
               ast.In: "in",
               ast.NotIn: "notin"}

supported_builtins = {"iter", "range", "enumerate", "zip", "all", "any", "max", "min", "abs", "pow",
                      "round", "reversed"}


def is_ellipsis(node):
    vi = sys.version_info
    if vi.major != 3 or vi.minor < 8:
        raise NotImplementedError("This only applies to CPython, version 3.8 and higher")
    else:
        # could patch in 7 by checking the deprecated node type.
        if isinstance(node, ast.Constant):
            return node.value == Ellipsis
    return False


def extract_positional_info(node):
    return ir.Position(line_begin=node.lineno,
                       line_end=node.end_lineno,
                       col_begin=node.col_offset,
                       col_end=node.end_col_offset)


class ImportHandler(ast.NodeVisitor):
    """
    This is used to replace source aliases with fully qualified names.

    """

    def __init__(self):
        self.import_map = None
        self.bound_names = None

    @contextmanager
    def make_context(self):
        self.bound_names = set()
        self.import_map = {}
        yield
        self.bound_names = None
        self.import_map = None

    def visit_Module(self, node):
        with self.make_context():
            for n in node.body:
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    self.visit(n)
            import_map = self.import_map
        return import_map

    def generic_visit(self, node):
        # ignore anything that isn't an import
        pass

    def visit_Import(self, node):
        # import modules only
        pos = extract_positional_info(node)
        for name in node.names:
            module_name = ir.NameRef(name.name)
            module_alias = ir.NameRef(name.asname) if hasattr(name, "asname") else module_name
            if module_alias in self.bound_names:
                msg = f"Module alias {module_alias.name} to module {module_name.name}" \
                      f" shadows a previously bound name."
                raise CompilerError(msg)
            mod_import = ir.ModImport(module_name, module_alias, pos)
            self.import_map[module_alias] = mod_import
            self.bound_names.add(module_alias)

    def visit_ImportFrom(self, node):
        module_name = ir.NameRef(node.module)
        pos = extract_positional_info(node)
        for name in node.names:
            # assume alias node
            imported_name = ir.NameRef(name.name)
            import_alias = ir.NameRef(name.asname) if hasattr(name, "asname") else imported_name
            if import_alias in self.bound_names:
                msg = "Name {import_alias} overwrites an existing assignment."
                raise CompilerError(msg)
            import_ref = ir.NameImport(module_name, imported_name, import_alias, pos)
            self.import_map[import_alias] = import_ref
            self.bound_names.add(import_alias)


class TreeBuilder(ast.NodeVisitor):

    def __init__(self):
        self.body = None
        self.entry = None
        self.enclosing_loop = None
        self.symbols = None
        self.renaming = None
        self.fold_if_constant = const_folding()

    @contextmanager
    def function_context(self, entry, symbols):
        self.enclosing_loop = None
        self.entry = entry
        self.symbols = symbols
        self.renaming = {}
        self.body = []
        yield
        assert self.enclosing_loop is None
        self.entry = None
        self.symbols = None
        self.renaming = None
        self.body = None

    def __call__(self, entry: ast.FunctionDef, symbols):
        with self.function_context(entry, symbols):
            func = self.visit(entry)
        return func

    @contextmanager
    def loop_region(self, header):
        stashed = self.body
        enclosing = self.enclosing_loop
        self.body = []
        self.enclosing_loop = header
        yield
        self.enclosing_loop = enclosing
        self.body = stashed

    @contextmanager
    def flow_region(self):
        stashed = self.body
        self.body = []
        yield
        self.body = stashed

    def is_local_variable_name(self, name: ir.NameRef):
        sym = self.symbols.lookup(name)
        if sym is None:
            return False
        return sym.is_source_name

    def visit_Attribute(self, node: ast.Attribute) -> ir.NameRef:
        value = self.visit(node.value)
        if isinstance(value, ir.NameRef):
            name = f"{value.name}.{node.attr}"
            value = ir.NameRef(name)
        else:
            msg = f"Attributes are only supported when attached to names, like 'module.function', received {node}."
            raise NotImplementedError(msg)
        return value

    def visit_Constant(self, node: ast.Constant) -> ir.Constant:
        if is_ellipsis(node.value):
            msg = "Ellipses are not supported."
            raise TypeError(msg)
        elif isinstance(node.value, str):
            # Check unicode code points fall within ascii range. This is mainly supported
            # for the possibility of enabling simple printing.
            if any(ord(v) > 127 for v in node.value):
                msg = f"Only strings that can be converted to ascii text are supported. This is mainly intended" \
                      f"to facilitate simple printing support at some point."
                raise CompilerError(msg)
            # special case, by default strings are wrapped as names
            # rather than constants.
            output = ir.StringConst(node.value)
        else:
            output = wrap_input(node.value)
        return output

    def visit_Tuple(self, node: ast.Tuple) -> ir.Tuple:
        # refactor seems to have gone wrong here..
        # version = sys.version_info
        # if (3, 9) <= (version.major, version.minor):
        # 3.9 removes ext_slice in favor of a tuple of slices
        # need to check a lot of cases for this
        #    raise NotImplementedError
        elts = tuple(self.visit(elt) for elt in node.elts)
        return ir.Tuple(elts)

    def visit_Name(self, node: ast.Name):
        return ir.NameRef(node.id)

    def visit_Expr(self, node: ast.Expr):
        # single expression statement
        expr = self.visit(node.value)
        pos = extract_positional_info(node)
        self.body.append(ir.SingleExpr(expr, pos))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ir.ValueRef:
        op = unary_ops.get(type(node.op))
        operand = self.visit(node.operand)
        if op == "+":  # This is a weird noop that can be ignored.
            expr = operand
        elif op == "not":
            expr = ir.NOT(operand)
        else:
            expr = ir.UnaryOp(operand, op)
        return expr

    def visit_BinOp(self, node: ast.BinOp) -> ir.BinOp:
        op = binary_ops.get(type(node.op))
        left = self.visit(node.left)
        left = self.fold_if_constant(left)
        right = self.visit(node.right)
        right = self.fold_if_constant(right)
        return ir.BinOp(left, right, op)

    def visit_BoolOp(self, node: ast.BoolOp) -> typing.Union[ir.BoolConst, ir.AND, ir.OR]:
        op = bool_ops[node.op]
        operands = []
        for value in node.values:
            value = self.visit(value)
            operands.append(value)
        operands = tuple(operands)
        if op == "and":
            expr = ir.AND(operands)
        else:
            expr = ir.OR(operands)
        return expr

    def visit_Compare(self, node: ast.Compare) -> typing.Union[ir.BinOp, ir.AND, ir.BoolConst]:
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        op = compare_ops[type(node.ops[0])]
        initial_compare = ir.CompareOp(left, right, op)
        initial_compare = self.fold_if_constant(initial_compare)
        if len(node.ops) == 1:
            return initial_compare
        compares = [initial_compare]
        for ast_op, comparator in zip(islice(node.ops, 1, None), islice(node.comparators, 1, None)):
            left = right
            right = self.visit(comparator)
            # bad operators will fail at AST construction
            op = compare_ops[type(ast_op)]
            cmp = ir.CompareOp(left, right, op)
            cmp = self.fold_if_constant(cmp)
            compares.append(cmp)
        return ir.AND(tuple(compares))

    def visit_Call(self, node: ast.Call) -> typing.Union[ir.ValueRef, ir.NameRef, ir.Call]:
        if isinstance(node.func, ast.Name):
            func_name = ir.NameRef(node.func.id)
        else:
            func_name = self.visit(node.func)
        args = tuple(self.visit(arg) for arg in node.args)
        keywords = tuple((kw.arg, self.visit(kw.value)) for kw in node.keywords)
        # replace call should handle folding of casts

        call_ = ir.Call(func_name, args, keywords)
        # Todo: need a way to identify array creation
        if not self.is_local_variable_name(func_name):
            # Most of the time this kind of conflict is an error. For example
            # we don't support overwriting zip, enumerate, etc. since we don't
            # really support customized iterator protocols anyway.
            call_ = replace_call(call_)
        return call_

    def visit_IfExp(self, node: ast.IfExp) -> ir.ValueRef:
        test = self.visit(node.test)
        on_true = self.visit(node.body)
        on_false = self.visit(node.orelse)
        expr = ir.Ternary(test, on_true, on_false)
        return expr

    def visit_Subscript(self, node: ast.Subscript) -> ir.Subscript:
        target = self.visit(node.value)
        s = self.visit(node.slice)
        value = ir.Subscript(target, s)
        return value

    def visit_Index(self, node: ast.Index) -> typing.Union[ir.ValueRef, ir.NameRef, ir.Constant]:
        return self.visit(node.value)

    def visit_Slice(self, node: ast.Slice) -> ir.Slice:
        lower = self.visit(node.lower) if node.lower is not None else None
        upper = self.visit(node.upper) if node.upper is not None else None
        step = self.visit(node.step) if node.step is not None else None
        return ir.Slice(lower, upper, step)

    def visit_ExtSlice(self, node: ast.ExtSlice):
        # This is probably never going to be supported, because it requires inlining
        # a large number of calculations in ways that may sometimes hamper performance.
        raise CompilerError("Extended slices are currently unsupported. This supports single"
                            "slices per dimension")

    def visit_AugAssign(self, node: ast.AugAssign):
        target = self.visit(node.target)
        operand = self.visit(node.value)
        op = binary_in_place_ops.get(type(node.op))
        pos = extract_positional_info(node)
        assign = ir.Assign(target, ir.BinOp(target, operand, op), pos)
        self.body.append(assign)

    def visit_Assign(self, node: ast.Assign):
        # first convert locally to internal IR
        value = self.visit(node.value)
        pos = extract_positional_info(node)
        for target in node.targets:
            # break cascaded assignments into multiple assignments
            target = self.visit(target)
            for subtarget, subvalue in unpack_assignment(target, value, pos):
                subassign = ir.Assign(subtarget, subvalue, pos)
                self.body.append(subassign)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        target = self.visit(node.target)
        value = self.visit(node.value)
        pos = extract_positional_info(node)
        annotation = self.visit(node.annotation)
        if isinstance(annotation, ir.NameRef):
            # Check if type is recognized by name
            type_ = self.symbols.type_by_name.get(annotation.name)
            if type_ is None:
                msg = f"Ignoring unrecognized annotation: {annotation}, line: {pos.line_begin}"
                warnings.warn(msg)
            else:
                ir_type = self.symbols.get_ir_type(type_)
                if isinstance(target, ir.NameRef):
                    sym = self.symbols.lookup(target)
                    existing_type = sym.type_
                    # This is an error, since it's an actual conflict.
                    if existing_type != ir_type:
                        msg = f"IR type from type hint conflicts with existing " \
                              f"(possibly inferred) type {existing_type}, line: {pos.line_begin}"
                        raise CompilerError(msg)
        if node.value is not None:
            # CPython will turn the syntax "var: annotation" into an AnnAssign node
            # with node.value = None. If executed, this won't bind or update the value of var.
            assign = ir.Assign(target, value, pos)
            self.body.append(assign)

    def visit_Pass(self, node: ast.Pass):
        # If required and missing, AST construction
        # will fail. After that these are somewhat useless.
        return

    def visit_If(self, node: ast.If):
        pos = extract_positional_info(node)
        compare = self.visit(node.test)
        with self.flow_region():
            for stmt in node.body:
                self.visit(stmt)
            on_true = self.body
        with self.flow_region():
            for stmt in node.orelse:
                self.visit(stmt)
            on_false = self.body
        ifstat = ir.IfElse(compare, on_true, on_false, pos)
        self.body.append(ifstat)

    def visit_For(self, node: ast.For):
        if node.orelse:
            raise CompilerError("or else clause not supported for for statements")
        iter_node = self.visit(node.iter)
        target_node = self.visit(node.target)
        assert iter_node is not None
        assert target_node is not None
        pos = extract_positional_info(node)
        targets = set()
        iterables = set()
        # Do initial checks for weird issues that may arise here.
        # We don't lower it fully at this point, because it injects
        # additional arithmetic and not all variable types may be fully known
        # at this point.
        try:
            for target, iterable in unpack_iterated(target_node, iter_node, include_enumerate_indices=True):
                targets.add(target)
                iterables.add(iterable)
        except ValueError:
            # Generator will throw an error on bad unpacking
            msg = f"Cannot safely unpack for loop expression, line: {pos.line_begin}"
            raise CompilerError(msg)
        conflicts = targets.intersection(iterables)
        if conflicts:
            conflict_names = ", ".join(c for c in conflicts)
            msg = f"{conflict_names} appear in both the target an iterable sequences of a for loop, " \
                  f"line {pos.line_begin}. This is not supported."
            raise CompilerError(msg)
        with self.loop_region(node):
            for stmt in node.body:
                self.visit(stmt)
            loop = ir.ForLoop(target_node, iter_node, self.body, pos)
        self.body.append(loop)

    def visit_While(self, node: ast.While):
        if node.orelse:
            raise CompilerError("or else clause not supported for for statements")
        test = self.visit(node.test)
        pos = extract_positional_info(node)
        with self.loop_region(node):
            for stmt in node.body:
                self.visit(stmt)
            loop = ir.WhileLoop(test, self.body, pos)
        self.body.append(loop)

    def visit_Break(self, node: ast.Break):
        if self.enclosing_loop is None:
            raise CompilerError("Break encountered outside of loop.")
        pos = extract_positional_info(node)
        stmt = ir.Break(pos)
        self.body.append(stmt)

    def visit_Continue(self, node: ast.Continue):
        if self.enclosing_loop is None:
            raise CompilerError("Continue encountered outside of loop.")
        pos = extract_positional_info(node)
        stmt = ir.Continue(pos)
        self.body.append(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # can't check just self.blocks, since we automatically 
        # instantiate entry and exit
        if node is not self.entry:
            raise CompilerError(f"Nested scopes are unsupported. line: {node.lineno}")
        name = node.name
        # Todo: warn about unsupported argument features
        params = [ir.NameRef(arg.arg) for arg in node.args.args]
        for stmt in node.body:
            self.visit(stmt)
        func = ir.Function(name, params, self.body)
        return func

    def visit_Return(self, node: ast.Return):
        pos = extract_positional_info(node)
        value = self.visit(node.value) if node.value is not None else None
        stmt = ir.Return(value, pos)
        self.body.append(stmt)

    def generic_visit(self, node):
        raise CompilerError(f"{type(node)} is unsupported")


def map_functions_by_name(node: ast.Module):
    by_name = {}
    assert(isinstance(node, ast.Module))
    for n in node.body:
        if isinstance(n, ast.FunctionDef):
            by_name[n.name] = n
    return by_name


def parse_file(file_name):
    """
    Module level point of entry for IR construction.

    Parameters
    ----------

    file_name:
        File path we used to extract source. This is used for error reporting.

    type_map:
        Map of function parameters and local variables to numpy or python numeric types.
   
    """

    path = Path(file_name)
    if not path.is_file():
        msg = f"Cannot resolve path to source {path.absolute()}"
        raise CompilerError(msg)

    with open(path) as src_stream:
        src_text = src_stream.read()
        syntax_tree = ast.parse(src_text, filename=path.name)
        syntax_tree = ast.fix_missing_locations(syntax_tree)
        import_map = ImportHandler().visit(syntax_tree)
        funcs_by_name = map_functions_by_name(syntax_tree)
        funcs = []
        module_symtable = symtable.symtable(src_text, file_name, "exec")
        build_func_ir = TreeBuilder()
        symbol_tables = {}
        for func_name, ast_entry_point in funcs_by_name.items():
            table = module_symtable.lookup(func_name).get_namespace()
            symbols = st_from_pyst(table, file_name)
            symbol_tables[func_name] = symbols
            func_ir = build_func_ir(ast_entry_point, symbols)
            funcs.append(func_ir)

    return ir.Module(funcs, import_map), symbol_tables
