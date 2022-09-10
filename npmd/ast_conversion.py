import ast
import sys
import typing
import symtable
import warnings

from contextlib import contextmanager
from itertools import islice
from pathlib import Path

import npmd.ir as ir

from npmd.canonicalize import replace_call
from npmd.errors import CompilerError
from npmd.folding import fold_constants, simplify_arithmetic
from npmd.pretty_printing import PrettyFormatter
from npmd.symbol_table import symbol, SymbolTable
from npmd.utils import unpack_iterated


binary_op_strs = {ast.Add: "+",
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

binary_ops = {ast.Add: ir.ADD,
              ast.Sub: ir.SUB,
              ast.Mult: ir.MULT,
              ast.Div: ir.TRUEDIV,
              ast.FloorDiv: ir.FLOORDIV,
              ast.Mod: ir.MOD,
              ast.Pow: ir.POW,
              ast.LShift: ir.LSHIFT,
              ast.RShift: ir.RSHIFT,
              ast.BitOr: ir.BITOR,
              ast.BitXor: ir.BITXOR,
              ast.BitAnd: ir.BITAND,
              ast.MatMult: ir.MATMULT}

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

unary_ops = {ast.USub: ir.USUB,
             ast.Invert: ir.UINVERT,
             ast.Not: ir.NOT}

bool_op_strs = {ast.And: "and",
                ast.Or: "or"}

bool_ops = {ast.And: ir.AND,
            ast.Or: ir.OR}

compare_op_strs = {ast.Eq: "==",
                   ast.NotEq: "!=",
                   ast.Lt: "<",
                   ast.LtE: "<=",
                   ast.Gt: ">",
                   ast.GtE: ">=",
                   ast.Is: "is",
                   ast.IsNot: "isnot",
                   ast.In: "in",
                   ast.NotIn: "notin"}

compare_ops = {ast.Eq: ir.EQ,
               ast.NotEq: ir.NE,
               ast.Lt: ir.LT,
               ast.LtE: ir.LE,
               ast.Gt: ir.GT,
               ast.GtE: ir.GE,
               ast.In: ir.IN,
               ast.NotIn: ir.NOTIN}

supported_builtins = {"iter", "range", "enumerate", "zip", "all", "any", "max", "min", "abs", "pow",
                      "round", "reversed"}


def unpack_assignment(target, value, pos):
    if isinstance(target, ir.TUPLE) and isinstance(value, ir.TUPLE):
        if target.length != value.length:
            msg = f"Cannot unpack {value} with {value.length} elements using {target} with {target.length} elements: " \
                  f"line {pos.line_begin}."
            raise ValueError(msg)
        for t, v in zip(target.subexprs, value.subexprs):
            yield from unpack_assignment(t, v, pos)
    else:
        yield target, value


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
        self.imports = None
        self.func_names = None

    @contextmanager
    def make_context(self):
        self.imports = {}
        self.func_names = []
        yield
        self.imports = None
        self.func_names = []

    def __call__(self, node):
        with self.make_context():
            self.visit(node)
            return self.imports, self.func_names

    def visit_Module(self, node):
        for n in node.body:
            self.visit(n)

    def generic_visit(self, node):
        # ignore anything that isn't an import
        pass

    class ImportHandler(ast.NodeVisitor):
        """
        This is used to replace source aliases with fully qualified names.

        """

        def __init__(self):
            self.imports = None
            self.numpy_alias = None

        @contextmanager
        def make_context(self):
            self.imports = {}
            yield
            self.imports = None
            self.numpy_alias = None

        def __call__(self, node):
            with self.make_context():
                self.visit(node)
                return self.imports, self.numpy_alias

        def visit_Module(self, node):
            for n in node.body:
                self.visit(n)

        def generic_visit(self, node):
            # ignore anything that isn't an import
            pass

        def visit_Import(self, node):
            # import modules only
            for name in node.names:
                if name.name != 'numpy':
                    names = ", ".join(name.name for name in node.names)
                    msg = f'Single import of numpy is the only currently supported import, received: {names}'
                    raise CompilerError(msg)
                self.numpy_alias = ir.NameRef(getattr(name, 'asname', 'numpy'))

        def visit_ImportFrom(self, node):
            module_name = node.module
            if module_name != 'numpy':
                msg = f'Only some imports from numpy are supported, received "{module_name}"'
                raise CompilerError(msg)
            for name in node.names:
                # assume alias node
                imported_name = name.name
                import_alias = getattr(name, 'asname', imported_name)
                if imported_name in self.imports:
                    msg = 'Name "{import_alias}" overwrites an existing assignment.'
                    raise CompilerError(msg)
                self.imports[ir.NameRef(import_alias)] = ir.NameRef(imported_name)


class TreeBuilder(ast.NodeVisitor):

    def __init__(self):
        self.symbols = None
        self.body = None
        self.entry = None
        self.enclosing_loop = None
        self.formatter = PrettyFormatter()

    @contextmanager
    def function_context(self, symbols):
        if self.symbols is not None:
            msg = f"Unexpected nested function context: {symbols.namespace}"
            raise RuntimeError(msg)
        self.symbols = symbols
        yield
        self.symbols = None

    @contextmanager
    def flow_region(self):
        stashed = self.body
        self.body = []
        yield
        self.body = stashed

    @contextmanager
    def loop_region(self, header):
        outer = self.enclosing_loop
        self.enclosing_loop = header
        with self.flow_region():
            yield
            self.enclosing_loop = outer

    def __call__(self, entry: ast.FunctionDef, symbols: SymbolTable):
        with self.function_context(symbols):
            func = self.visit(entry)
        return func

    def visit_Attribute(self, node: ast.Attribute):
        v = self.visit(node.value)
        if isinstance(v, ir.NameRef):
            return ir.NameRef(f"{v.name}.{node.attr}")
        else:
            formatted = self.formatter(node.value)
            msg = f"Attributes are disallowed on everything except name references. Received: {formatted}."
            raise CompilerError(msg)

    def visit_Constant(self, node: ast.Constant) -> ir.CONSTANT:
        if is_ellipsis(node.value):
            msg = "Ellipses are not supported."
            raise TypeError(msg)
        elif isinstance(node.value, str):
            # This will check that text is convertible to ascii.
            output = ir.StringConst(node.value)
        else:
            output = ir.wrap_constant(node.value)
        return output

    def visit_Tuple(self, node: ast.Tuple) -> ir.TUPLE:
        # refactor seems to have gone wrong here..
        # version = sys.version_info
        # if (3, 9) <= (version.major, version.minor):
        # 3.9 removes ext_slice in favor of a tuple of slices
        # need to check a lot of cases for this
        #    raise NotImplementedError
        return ir.TUPLE(*(self.visit(elt) for elt in node.elts))

    def visit_Name(self, node: ast.Name):
        return ir.NameRef(node.id)

    def visit_Expr(self, node: ast.Expr):
        # single expression statement
        expr = self.visit(node.value)
        pos = extract_positional_info(node)
        # don't append single value references as statements
        if not isinstance(expr, (ir.CONSTANT, ir.NameRef, ir.Subscript, ir.StringConst)):
            self.body.append(ir.SingleExpr(expr, pos))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ir.ValueRef:
        if node.op == "+":
            expr = self.visit(node.operand)
        else:
            cls = unary_ops.get(type(node.op))
            operand = self.visit(node.operand)
            expr = cls(operand)
        expr = fold_constants(expr)
        expr = simplify_arithmetic(expr)
        return expr

    def visit_BinOp(self, node: ast.BinOp) -> ir.BinOp:
        cls = binary_ops.get(type(node.op))
        left = self.visit(node.left)
        right = self.visit(node.right)
        expr = cls(left, right)
        expr = fold_constants(expr)
        expr = simplify_arithmetic(expr)
        return expr

    def visit_BoolOp(self, node: ast.BoolOp) -> typing.Union[ir.CONSTANT, ir.AND, ir.OR]:
        cls = bool_ops.get(node.op)
        expr = cls(*(self.visit(v) for v in node.values))
        expr = fold_constants(expr)
        expr = simplify_arithmetic(expr)
        return expr

    def visit_Compare(self, node: ast.Compare) -> typing.Union[ir.BinOp, ir.AND, ir.CONSTANT]:
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        cls = compare_ops[type(node.ops[0])]
        initial_compare = cls(left, right)
        if len(node.ops) == 1:
            return initial_compare
        compares = [initial_compare]
        for ast_op, comparator in zip(islice(node.ops, 1, None), islice(node.comparators, 1, None)):
            left = right
            right = self.visit(comparator)
            # bad operators will fail at AST construction
            cls = compare_ops[type(ast_op)]
            cmp = cls(left, right)
            compares.append(cmp)
        return ir.AND(*compares)

    def visit_Call(self, node: ast.Call) -> typing.Union[ir.ValueRef, ir.NameRef, ir.Call]:
        if not isinstance(node.func, ast.Name):
            raise CompilerError('Calls only supported on function names.')
        if isinstance(node.func, ast.Name):
            func_name = ir.NameRef(node.func.id)
            if node.keywords:
                # Todo: add AST positional arguments, since these are available for expressions
                #  at Python AST level
                msg = 'Keyword arguments are unsupported.'
                raise CompilerError(msg)
            args = [func_name]
            for arg in node.args:
                args.append(self.visit(arg))
            call_ = ir.Call(*args)
            # Todo: need a way to identify array creation
            repl = replace_call(call_)
            return repl
        else:
            no_match_msg = f'No matching supported signature for "{node}". ' \
                           f'Non-default arguments not yet supported on reductions.'
            arg = self.visit(node.func.value)
            if not isinstance(node.func, ast.Attribute):
                msg = f'Unsupported call "{node.func}". Only names and some attributes are supported as callable.'
                raise CompilerError(msg)
            elif len(node.keywords) or len(node.args):
                raise CompilerError(no_match_msg)
            if node.func.attr == 'min':
                return ir.ARRAY_MIN(arg)
            elif node.func.attr == 'max':
                return ir.ARRAY_MAX(arg)
            elif node.func.attr == 'sum':
                return ir.ARRAY_SUM(arg)
            else:
                raise CompilerError(no_match_msg)

    def visit_IfExp(self, node: ast.IfExp) -> ir.ValueRef:
        predicate = self.visit(node.test)
        on_true = self.visit(node.body)
        on_false = self.visit(node.orelse)
        expr = ir.SELECT(on_true, on_false, predicate)
        return expr

    def visit_Subscript(self, node: ast.Subscript) -> typing.Union[ir.Subscript, ir.Slice]:
        target = self.visit(node.value)
        s = self.visit(node.slice)
        if isinstance(s, ir.Slice):
            if all(isinstance(sl, ir.NoneRef) for sl in s.subexprs):
                return target
            # sanitize any Nones so that we don't need the full subscript to interpret this
            if any(isinstance(sl, ir.NoneRef) for sl in s.subexprs):
                start, stop, step = s.subexprs
                if isinstance(start, ir.NoneRef):
                    start = ir.Zero
                if isinstance(stop, ir.NoneRef):
                    stop = ir.SingleDimRef(target, ir.Zero)
                if isinstance(step, ir.NoneRef):
                    step = ir.One
                s = ir.Slice(start, stop, step)
        return ir.Subscript(target, s)

    def visit_Index(self, node: ast.Index) -> typing.Union[ir.ValueRef, ir.NameRef, ir.CONSTANT]:
        return self.visit(node.value)

    def visit_Slice(self, node: ast.Slice) -> ir.Slice:
        lower = self.visit(node.lower) if node.lower is not None else ir.NoneRef()
        upper = self.visit(node.upper) if node.upper is not None else ir.NoneRef()
        step = self.visit(node.step) if node.step is not None else ir.NoneRef()
        return ir.Slice(lower, upper, step)

    def visit_ExtSlice(self, node: ast.ExtSlice):
        # This is probably never going to be supported, because it requires inlining
        # a large number of calculations in ways that may sometimes hamper performance.
        raise CompilerError('Extended slices are currently unsupported. This supports single'
                            'slices per dimension')

    def visit_AugAssign(self, node: ast.AugAssign):
        target = self.visit(node.target)
        operand = self.visit(node.value)
        cls = binary_ops[type(node.op)]
        pos = extract_positional_info(node)
        expr = cls(target, operand)
        assign = ir.InPlaceOp(target, expr, pos)
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
                msg = f'Ignoring unrecognized annotation: {annotation}, line: {pos.line_begin}'
                warnings.warn(msg)
            else:
                ir_type = self.symbols.get_ir_type(type_)
                if isinstance(target, ir.NameRef):
                    sym = self.symbols.lookup(target)
                    existing_type = sym.type_
                    # This is an error, since it's an actual conflict.
                    if existing_type != ir_type:
                        msg = f'IR type from type hint conflicts with existing ' \
                              f'(possibly inferred) type {existing_type}, line: {pos.line_begin}'
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
            raise CompilerError('or else clause not supported for for statements')
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
            for target, iterable in unpack_iterated(target_node, iter_node):
                # check for some things that aren't currently supported
                if isinstance(target, ir.Subscript):
                    msg = f'Setting a subscript target in a for loop iterator is currently unsupported: line{pos.line_begin}.'
                    raise CompilerError(msg)
                targets.add(target)
                iterables.add(iterable)
        except ValueError:
            # Generator will throw an error on bad unpacking
            msg = f'Cannot safely unpack for loop expression, line: {pos.line_begin}'
            raise CompilerError(msg)
        conflicts = targets.intersection(iterables)
        if conflicts:
            conflict_names = ", ".join(c for c in conflicts)
            msg = f'{conflict_names} appear in both the target an iterable sequences of a for loop, ' \
                  f'line {pos.line_begin}. This is not supported.'
            raise CompilerError(msg)
        with self.loop_region(node):
            for stmt in node.body:
                self.visit(stmt)
            loop = ir.ForLoop(target_node, iter_node, self.body, pos)
        self.body.append(loop)

    def visit_While(self, node: ast.While):
        if node.orelse:
            raise CompilerError('or else clause not supported for for statements')
        test = self.visit(node.test)
        pos = extract_positional_info(node)
        with self.loop_region(node):
            for stmt in node.body:
                self.visit(stmt)
            loop = ir.WhileLoop(test, self.body, pos)
        self.body.append(loop)

    def visit_Break(self, node: ast.Break):
        if self.enclosing_loop is None:
            raise CompilerError('Break encountered outside of loop.')
        pos = extract_positional_info(node)
        stmt = ir.Break(pos)
        self.body.append(stmt)

    def visit_Continue(self, node: ast.Continue):
        if self.enclosing_loop is None:
            raise CompilerError('Continue encountered outside of loop.')
        pos = extract_positional_info(node)
        stmt = ir.Continue(pos)
        self.body.append(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # can't check just self.blocks, since we automatically 
        # instantiate entry and exit
        name = node.name
        # Todo: warn about unsupported argument features
        params = [ir.NameRef(arg.arg) for arg in node.args.args]
        with self.flow_region():
            for stmt in node.body:
                self.visit(stmt)
            func = ir.Function(name, params, self.body)
        return func

    def visit_Return(self, node: ast.Return):
        pos = extract_positional_info(node)
        value = self.visit(node.value) if node.value is not None else ir.NoneRef()
        stmt = ir.Return(value, pos)
        self.body.append(stmt)

    def generic_visit(self, node):
        raise CompilerError(f'{type(node)} is unsupported')


def map_functions_by_name(node: ast.Module):
    by_name = {}
    assert (isinstance(node, ast.Module))
    for n in node.body:
        if isinstance(n, ast.FunctionDef):
            by_name[n.name] = n
    return by_name


def populate_func_symbols(func_table, types, allow_inference=True, ignore_unbound=False):
    """
    Turns python symbol table into a typed internal representation.

    """
    # Todo: This needs a better way to deal with imports where they are used.
    # since we have basic type info here, later phases should be validating the argument
    # and type signatures used when calling imported functions
    if allow_inference and ignore_unbound:
        msg = 'Cannot allow type inference if unbound variable paths are allowed here.'
        raise CompilerError(msg)
    func_name = func_table.get_name()
    symbols = {}
    missing = []
    missing_arg_types = []
    for s in func_table.get_symbols():
        if s.is_parameter():
            type_ = types.get(s.get_name())
            if type_ is None:
                missing_arg_types.append(s.get_name())
    if missing_arg_types:
        missing_arg_types = ', '.join(s for s in missing_arg_types)
        msg = f'In function: "{func_name}", missing type information for arguments "{missing_arg_types}"'
        raise CompilerError(msg)
    for s in func_table.get_symbols():
        name = s.get_name()
        if s.is_imported():
            msg = 'Locally importing names within a function is not currently supported: ' \
                  f'Function: {func_name} , imports: {name}'
            raise CompilerError(msg)
        is_arg = s.is_parameter()
        is_assigned = s.is_assigned()
        is_local = s.is_local()
        if is_assigned and not is_local:
            msg = f'Assigning to global and non-local variables is unsupported: {name} in function {func_name}'
            raise CompilerError(msg)
        elif is_local:
            if not (ignore_unbound or is_assigned or is_arg):
                msg = f'Local variable {name} in function {func_name} is unassigned. ' \
                      f'This is automatically treated as an error.'
                raise CompilerError(msg)
            type_ = types.get(name)
            if type_ is None:
                missing.append(name)
            sym = symbol(name,
                         type_,
                         is_arg,
                         is_source_name=True,
                         is_local=is_local,
                         is_assigned=is_assigned,
                         type_is_inferred=False)
            symbols[name] = sym
        elif name in types:
            # if a type is given here but this is not local, raise an error
            if s.is_referenced():
                msg = f'Typed name {name} does not match an assignment.'
                raise CompilerError(msg)
    if missing and not allow_inference:
        msg = f'No type provided for local variable(s): {missing}'
        raise CompilerError(msg)
    return SymbolTable(func_name, symbols)


def populate_symbol_tables(module_name, src, types):
    table = symtable.symtable(code=src, filename=module_name, compile_type="exec")

    func_names = set()
    imports = set()
    func_tables = {}

    # check imports
    for sym in table.get_symbols():
        name = sym.get_name()
        if sym in imports:
            msg = f"symbol {name} shadows an existing entry."
            raise CompilerError(msg)
        elif sym.is_imported():
            imports.add(name)

    for func_table in table.get_children():
        t = func_table.get_type()
        name = func_table.get_name()
        if t == "class":
            msg = f"classes are unsupported: {name}"
            raise CompilerError(msg)
        elif name in imports:
            msg = f"Namespace {name} shadows an existing entry."
            raise CompilerError(msg)
        elif func_table.has_children():
            msg = f"Function {name} contains nested namespaces, which are unsupported."
            raise CompilerError(msg)
        func_names.add(name)
        func_types = types.get(name, dict())
        # declare a header file to avoid worrying about declaration order
        func_tables[name] = populate_func_symbols(func_table, func_types, allow_inference=True, ignore_unbound=False)
    return func_tables


def build_module_ir_and_symbols(src_path, types):
    path = Path(src_path)
    module_name = path.name
    src_text = path.read_text()
    func_symbol_tables = populate_symbol_tables(module_name, src_text, types)
    mod_ast = ast.parse(src_text, filename=path.name)
    mod_ast = ast.fix_missing_locations(mod_ast)
    import_handler = ImportHandler()
    import_map = import_handler(mod_ast)
    funcs_by_name = map_functions_by_name(mod_ast)
    build_ir = TreeBuilder()
    funcs = []
    # Todo: register imports once interface stabilizes
    for func_name, ast_entry_point in funcs_by_name.items():
        symbol_handler = func_symbol_tables.get(func_name)
        func_ir = build_ir(ast_entry_point, symbol_handler)
        funcs.append(func_ir)
    return ir.Module(module_name, funcs, import_map), func_symbol_tables