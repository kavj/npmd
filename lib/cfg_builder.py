import ast
import inspect
import itertools
import operator
import sys
import typing
import symtable

from collections import defaultdict
from contextlib import contextmanager
from itertools import islice
from pathlib import Path

from typing import Dict, Tuple

import lib.ir as ir
import networkx as nx

from lib.blocks import BasicBlock, FunctionContext
from lib.callspec import early_call_specialize
from lib.errors import CompilerError
from lib.formatting import PrettyFormatter
from lib.walkers import walk_expr, walk_graph
from lib.symbol_table import symbol, SymbolTable


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
    This is used to map a bound name to a module or import method

    """

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.imports = {}
        self.modules = {}

    @classmethod
    def collect_imports(cls, node: ast.Module, modname: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        visitor = cls(modname)
        visitor.visit(node)
        return visitor.modules, visitor.imports

    def visit_Module(self, node: ast.Module):
        for n in node.body:
            self.visit(n)

    def generic_visit(self, node):
        # ignore anything that isn't an import
        pass

    def visit_Import(self, node):
        # import modules only
        for name in node.names:
            if name.name.endswith(self.module_name):
                msg = f'Imported name {name} conflicts with module name {self.module_name}'
                raise CompilerError(msg)
            if name.name != 'numpy':
                names = ", ".join(name.name for name in node.names)
                msg = f'Single import of numpy is the only currently supported import, received: {names}'
                raise CompilerError(msg)
            alias = getattr(name, 'asname', name)
            self.modules[alias] = name.name

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module_name = node.module
        if module_name != 'numpy':
            msg = f'Only some imports from numpy are supported, received "{module_name}"'
            raise CompilerError(msg)
        for name in node.names:
            # assume alias node
            imported_name = name.name
            import_alias = getattr(name, 'asname', imported_name)
            if name in itertools.chain(self.imports.keys(), self.imports.values()) \
                    or import_alias in self.imports.keys() \
                    or import_alias in self.modules.keys():
                msg = f'Name "{imported_name}" imported as "{import_alias}" conflicts with an existing import entry.'
                raise CompilerError(msg)
            if not imported_name.startswith('numpy'):
                msg = f'This currently only supports importing a subset of Numpy and no other libraries. ' \
                      f'Received import statement "{imported_name}".'
                raise CompilerError(msg)
            qualname = f'{module_name}.{imported_name}'
            self.imports[import_alias] = qualname


def resolve_call_target(node: ast.Call, module_imports: Dict[str, str], name_imports: Dict[str, str]):
    name = node.func
    attr_chain = []
    while isinstance(name, ast.Attribute):
        attr_chain.append(name.attr)
        name = name.value
    # no calling subscripted
    assert isinstance(name, ast.Name)
    attr_chain.append(name.id)
    attr_chain.reverse()
    base = attr_chain[0]
    if len(attr_chain) == 1 and base in module_imports:
        target = module_imports[base]
        msg = f'Module import "{target}" with alias "{base}" is not directly callable.'
        raise CompilerError(msg)
    elif base in module_imports:
        attr_chain[0] = module_imports[base]
        qualname = '.'.join(attr_chain)
    elif base in name_imports:
        attr_chain[0] = name_imports[base]
        qualname = '.'.join(attr_chain)
    else:
        for index, term in enumerate(itertools.islice(attr_chain, 1, None), 1):
            base = f'{base}.{term}'
            if base in module_imports:
                attr_chain[0] = module_imports[base]
                break
            elif base in name_imports:
                attr_chain[0] = name_imports[base]
                break
        qualname = '.'.join(attr_chain)
    return qualname


def find_graph_entry_points(graph: nx.DiGraph):
    for n in walk_graph(graph):
        if n.is_function_entry:
            return n


class BuilderContext:
    """
    Single use context holder
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.counter = itertools.count()
        self.loops = []
        self.annotations = defaultdict(list)
        self.entry_block = None
        self.current_block = None
        self.depth = 0

    def finalize(self):
        assert not self.loops
        assert self.depth == 0
        assert operator.truth(self.graph)
        entry_block = find_graph_entry_points(self.graph)
        assert entry_block is not None
        graph = self.graph
        counter = self.counter
        annotations = self.annotations
        self.graph = None
        self.counter = None
        self.annotations = None
        return FunctionContext(graph, entry_block, labeler=counter), annotations

    @contextmanager
    def loop_region(self):
        self.enter_loop()
        yield
        self.exit_loop()

    @property
    def next_label(self):
        return next(self.counter)

    @property
    def terminated(self):
        return self.current_block.terminated

    @property
    def unterminated(self):
        return self.current_block.unterminated

    def create_block(self):
        label = self.next_label
        block = BasicBlock([], label, self.depth)
        return block

    def append_statement(self, stmt: ir.Statement):
        assert isinstance(stmt, (ir.StmtBase, ir.Function))
        self.current_block.append_statement(stmt)

    def add_edge(self, u: BasicBlock, v: BasicBlock):
        self.graph.add_edge(u, v)

    def add_block(self, parent: typing.Optional[BasicBlock] = None):
        label = next(self.counter)
        block = BasicBlock([], label, self.depth)
        if parent is not None:
            assert isinstance(parent, BasicBlock)
            assert isinstance(block, BasicBlock)
            self.graph.add_edge(parent, block)
        else:
            self.graph.add_node(block)
        self.current_block = block
        return block

    def append_block(self):
        label = next(self.counter)
        block = BasicBlock([], label, self.depth)
        if self.current_block is not None and self.current_block.unterminated:
            self.add_edge(self.current_block, block)
        self.current_block = block
        return block

    def enter_loop(self):
        header = self.current_block
        self.depth += 1
        self.loops.append((header, []))
        body = self.append_block()
        self.current_block = body
        return body

    def exit_loop(self):
        assert self.depth > 0
        if self.current_block.unterminated:
            self.add_back_edge()
        header, break_blocks = self.loops.pop()
        self.depth -= 1
        exit_block = BasicBlock([], self.next_label, self.depth)
        self.add_edge(header, exit_block)
        for b in break_blocks:
            self.add_edge(b, exit_block)
        self.current_block = exit_block

    def add_back_edge(self):
        self.add_edge(self.current_block, self.loops[-1][0])

    def add_break(self):
        self.loops[-1][1].append(self.current_block)

    def predecessors(self, block: typing.Optional[BasicBlock] = None):
        """
        debugging helper routine
        :param block:
        :return:
        """
        if block is None:
            block = self.current_block
        if block is None:
            return []
        return list(self.graph.predecessors(block))


class CFGBuilder(ast.NodeVisitor):

    def __init__(self, name_imports: Dict[str, str], module_imports: Dict[str, str]):
        self.name_imports = name_imports
        self.module_imports = module_imports
        self.formatter = PrettyFormatter()
        self.builder_ctx = None

    @contextmanager
    def function_scope(self):
        assert self.builder_ctx is None
        self.builder_ctx = BuilderContext()
        yield
        self.builder_ctx = None

    def __call__(self, node: ast.FunctionDef):
        with self.function_scope():
            self.visit_FunctionDef(node)
            return self.builder_ctx.finalize()

    def visit_Attribute(self, node: ast.Attribute):
        base = node.value
        stack = [node]
        while isinstance(base, ast.Attribute):
            base = base.value
            stack.append(base)
        attributes = [self.visit(base)]
        for s in reversed(stack):
            attributes.append(ir.NameRef(s.attr))
        return ir.AttributeRef(attributes)

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
        # don't append single value references with no side effects as statements
        if any(isinstance(e, ir.Call) for e in walk_expr(expr)):
            self.builder_ctx.append_statement(ir.SingleExpr(expr, pos))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ir.ValueRef:
        if node.op == "+":
            expr = self.visit(node.operand)
        else:
            cls = unary_ops.get(type(node.op))
            operand = self.visit(node.operand)
            expr = cls(operand)
        return expr

    def visit_BinOp(self, node: ast.BinOp) -> ir.BinOp:
        cls = binary_ops.get(type(node.op))
        left = self.visit(node.left)
        right = self.visit(node.right)
        expr = cls(left, right)
        return expr

    def visit_BoolOp(self, node: ast.BoolOp) -> typing.Union[ir.CONSTANT, ir.AND, ir.OR]:
        cls = bool_ops.get(node.op)
        expr = cls(*(self.visit(v) for v in node.values))
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
        call_name = resolve_call_target(node, self.module_imports, self.name_imports)
        args = ir.TUPLE(*(self.visit(arg) for arg in node.args))
        # Todo: need a way to identify array creation
        if node.keywords:
            kwargs = []
            for kw in node.keywords:
                name = ir.NameRef(kw.arg)
                value = self.visit(kw.value)
                kwargs.append(ir.TUPLE(name, value))
            keywords = ir.TUPLE(*kwargs)
        else:
            keywords = ir.TUPLE()
        # Now check for aliasing
        call_ = ir.Call(call_name, args, keywords)
        repl = early_call_specialize(call_)
        return repl

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
        self.builder_ctx.append_statement(assign)

    def visit_Assign(self, node: ast.Assign):
        # first convert locally to internal IR
        value = self.visit(node.value)
        pos = extract_positional_info(node)
        for target in node.targets:
            # break cascaded assignments into multiple assignments
            target = self.visit(target)
            for subtarget, subvalue in unpack_assignment(target, value, pos):
                subassign = ir.Assign(subtarget, subvalue, pos)
                self.builder_ctx.append_statement(subassign)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        target = self.visit(node.target)
        pos = extract_positional_info(node)
        annotation = self.visit(node.annotation)
        self.builder_ctx.annotations[target].append(annotation)
        if node.value is not None:
            value = self.visit(node.value)
            assign = ir.Assign(target, value, pos)
            self.builder_ctx.append_statement(assign)

    def visit_Pass(self, node: ast.Pass):
        # If required and missing, AST construction
        # will fail. After that these are somewhat useless.
        return

    def visit_If(self, node: ast.If):
        pos = extract_positional_info(node)
        test = self.visit(node.test)
        # make incoming edge
        branch_block = self.builder_ctx.current_block
        if branch_block:
            # already contains statements
            branch_block = self.builder_ctx.append_block()
        if_block = self.builder_ctx.append_block()
        for stmt in node.body:
            self.visit(stmt)
            if self.builder_ctx.current_block.terminated:
                # don't follow break, continue, or return in this statement list
                break
        last_if_block = self.builder_ctx.current_block
        else_block = self.builder_ctx.add_block(parent=branch_block)
        for stmt in node.orelse:
            self.visit(stmt)
            if self.builder_ctx.current_block.terminated:
                # don't follow break, continue, or return in this statement list
                break
        last_else_block = self.builder_ctx.current_block
        exit_block = self.builder_ctx.add_block()
        for block in (last_if_block, last_else_block):
            if block.unterminated:
                self.builder_ctx.graph.add_edge(block, exit_block)

        # Now make a branch statement
        branch_stmt = ir.IfElse(test,
                                if_block.label,
                                else_block.label,
                                pos)
        branch_block.append_statement(branch_stmt)
        self.builder_ctx.current_block = exit_block

    def visit_For(self, node: ast.For):
        if node.orelse:
            raise CompilerError('or else clause not supported for for statements')
        if self.builder_ctx.current_block:
            self.builder_ctx.append_block()
        header_block = self.builder_ctx.current_block
        iter_node = self.visit(node.iter)
        target_node = self.visit(node.target)
        assert iter_node is not None
        assert target_node is not None
        pos = extract_positional_info(node)
        with self.builder_ctx.loop_region():
            for stmt in node.body:
                if isinstance(stmt, ast.Continue):
                    # if we see a continue in this statement list,
                    # we can terminate the loop body here without adding it
                    break
                self.visit(stmt)
                if self.builder_ctx.terminated:
                    # don't follow break, continue, or return in this statement list
                    break
        header = ir.ForLoop(target_node, iter_node, pos)
        header_block.append_statement(header)

    def visit_While(self, node: ast.While):
        if node.orelse:
            raise CompilerError('or else clause not supported for for statements')
        test = self.visit(node.test)
        pos = extract_positional_info(node)
        if self.builder_ctx.current_block:
            self.builder_ctx.append_block()
        header_block = self.builder_ctx.current_block
        with self.builder_ctx.loop_region():
            for stmt in node.body:
                self.visit(stmt)
                if self.builder_ctx.terminated:
                    # don't follow break, continue, or return in this statement list
                    break
        header = ir.WhileLoop(test, pos)
        header_block.append_statement(header)

    def visit_Break(self, node: ast.Break):
        pos = extract_positional_info(node)
        stmt = ir.Break(pos)
        self.builder_ctx.append_statement(stmt)
        self.builder_ctx.add_break()

    def visit_Continue(self, node: ast.Continue):
        pos = extract_positional_info(node)
        stmt = ir.Continue(pos)
        self.builder_ctx.append_statement(stmt)
        self.builder_ctx.add_back_edge()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # can't check just self.blocks, since we automatically 
        # instantiate entry and exit
        name = node.name
        # Todo: warn about unsupported argument features
        params = tuple(ir.NameRef(arg.arg) for arg in node.args.args)
        # docstrings show up as a SingleExpr node, which must be explicitly removed later.
        docstring = ast.get_docstring(node)
        header = ir.Function(name, params, docstring)
        self.builder_ctx.add_block()
        self.builder_ctx.append_statement(header)
        self.builder_ctx.append_block()
        for stmt in node.body:
            self.visit(stmt)
            if self.builder_ctx.terminated:
                # don't follow break, continue, or return in this statement list
                break
        if self.builder_ctx.unterminated:
            pos = ir.Position(-1, -1, 1, 40)
            stmt = ir.Return(ir.NoneRef(), pos)
            self.builder_ctx.append_statement(stmt)

    def visit_Return(self, node: ast.Return):
        pos = extract_positional_info(node)
        value = self.visit(node.value) if node.value is not None else ir.NoneRef()
        stmt = ir.Return(value, pos)
        self.builder_ctx.append_statement(stmt)

    def generic_visit(self, node):
        raise CompilerError(f'{type(node)} is unsupported')


def map_functions_by_name(node: ast.Module):
    by_name = {}
    assert (isinstance(node, ast.Module))
    for n in node.body:
        if isinstance(n, ast.FunctionDef):
            by_name[n.name] = n
    return by_name


def build_function_symbol_table(func_table: symtable.SymbolTable, types, ignore_unbound=False):
    """
    Turns python symbol table into a typed internal representation.

    """
    # Todo: This needs a better way to deal with imports where they are used.
    # since we have basic type info here, later phases should be validating the argument
    # and type signatures used when calling imported functions

    func_name = func_table.get_name()
    symbols = {}
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
        func_tables[name] = build_function_symbol_table(func_table, func_types, ignore_unbound=True)
    return func_tables


def build_ir_from_func(func):
    if not inspect.isfunction(func):
        msg = f'Expected an imported function. Received: {func}'
        raise CompilerError(msg)
    src_path = inspect.getfile(func)
    module_name = Path(src_path).name
    src_text = inspect.getsource(func)
    mod_ast = ast.parse(src_text, filename=module_name)
    funcs_by_name = map_functions_by_name(mod_ast)
    for func_name, func_ast in funcs_by_name.items():
        if func_name == func.__name__:
            builder = CFGBuilder({}, {})
            func_ctx, annotations = builder(func_ast)
            func_symbol_tables = populate_symbol_tables(module_name, src_text, {})
            symbols = func_symbol_tables[func_name]
            func_ctx.symbols = symbols
            return func_ctx


def build_module_ir(src_path, types):
    path = Path(src_path)
    module_name = path.name
    src_text = path.read_text()
    func_symbol_tables = populate_symbol_tables(module_name, src_text, types)
    mod_ast = ast.parse(src_text, filename=path.name)
    mod_ast = ast.fix_missing_locations(mod_ast)
    module_imports, name_imports = ImportHandler.collect_imports(mod_ast, module_name)
    funcs_by_name = map_functions_by_name(mod_ast)
    builder = CFGBuilder(module_imports, name_imports)
    funcs = []
    # Todo: register imports once interface stabilizes
    for func_name, func_ast in funcs_by_name.items():
        symbol_handler = func_symbol_tables.get(func_name)
        func_ctx, annotations = builder(func_ast)
        func_ctx.symbols = symbol_handler
        funcs.append(func_ctx)
    return ir.Module(module_name, funcs, module_imports, name_imports)
