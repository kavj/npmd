import ast
import operator
import sys
import typing
import symtable
import warnings

from contextlib import contextmanager

import ir
from symbol_table import symbol_table_from_pysymtable, wrap_input
from Canonicalize import replace_builtin_call
from lowering import const_folding, make_loop_counter

binaryops = {ast.Add: "+",
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

binary_inplace_ops = {ast.Add: "+=",
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

unaryops = {ast.UAdd: "+",
            ast.USub: "-",
            ast.Invert: "~",
            ast.Not: "not"}

boolops = {ast.And: "and",
           ast.Or: "or"}

compareops = {ast.Eq: "==",
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


def unpackable_length(iterable: ir.Call):
    func_name = iterable.func
    if func_name.name == "zip":
        assert len(iterable.keywords) == 0
        return len(iterable.args)
    elif func_name.name == "enumerate":
        return 2
    else:
        return 0


def unpack_iterated(target, iterable, pos):
    if isinstance(iterable, ir.Zip):
        # must unpack
        if isinstance(target, ir.Tuple):
            if len(target.elements) == len(iterable.elements):
                for t, v in zip(target.elements, iterable.elements):
                    yield from unpack_iterated(t, v, pos)
            else:
                msg = f"Mismatched unpacking counts for {target} and {iterable}, {len(target.elements)} " \
                      f"and {(len(iterable.elements))}."
                raise ValueError(msg)
        else:
            msg = f"Zip construct {iterable} requires a tuple for unpacking."
            raise ValueError(msg)

    else:
        # Array or sequence reference, with a single opaque target.
        yield target, iterable


def unpack_assignment(target, value, pos):
    if isinstance(target, ir.Tuple) and isinstance(value, ir.Tuple):
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


def extract_loop_indexer(node: ast.For):
    """
    Find cases of

    for index_name, values in enumerate(...):
        ...

    Here we can reuse index_name as a loop index prefix, regardless
    of clobbering.

    """

    target = node.target
    iterable = node.iter
    if isinstance(target, ast.Name) and isinstance(iterable, ast.Call):
        if isinstance(iterable.func, ast.Name):
            if iterable.func.id == "enumerate":
                return target.id


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
        imported_name = None
        for name in node.names:
            module_name = ir.NameRef(name.name)
            module_alias = ir.NameRef(name.asname) if hasattr(name, "asname") else None
            bound_name = module_alias if module_alias is not None else module_name
            if bound_name in self.bound_names:
                if module_alias is None:
                    msg = f"Module name {module_name.name} shadows a previously bound name."
                else:
                    msg = f"Module alias {module_alias.name} to module {module_name.name}" \
                          f" shadows a previously bound name."
                raise ValueError(msg)
            import_ref = ir.ImportRef(module_name, imported_name, module_alias)
            self.import_map[bound_name] = import_ref
            self.bound_names.add(bound_name)

    def visit_ImportFrom(self, node):
        module_name = ir.NameRef(node.module)
        for name in node.names:
            # assume alias node
            imported_name = ir.NameRef(name.name)
            import_alias = ir.NameRef(name.asname) if hasattr(name, "asname") else None
            bound_name = import_alias if import_alias is not None else imported_name
            if bound_name in self.bound_names:
                msg = "Name {import_alias} overwrites an existing assignment."
                raise ValueError(msg)
            import_ref = ir.ImportRef(module_name, imported_name, import_alias)
            self.import_map[bound_name] = import_ref
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
        name = name.name
        return name in self.symbols.src_locals

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
            msg = "String constants are not supported."
            raise TypeError(msg)
        return wrap_input(node.value)

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

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ir.UnaryOp:
        op = unaryops.get(type(node.op))
        operand = self.visit(node.operand)
        operand = self.fold_if_constant(operand)
        return ir.UnaryOp(operand, op) if op != "+" else operand

    def visit_BinOp(self, node: ast.BinOp) -> ir.BinOp:
        op = binaryops.get(type(node.op))
        left = self.visit(node.left)
        left = self.fold_if_constant(left)
        right = self.visit(node.right)
        right = self.fold_if_constant(right)
        return ir.BinOp(left, right, op)

    def visit_BoolOp(self, node: ast.BoolOp) -> typing.Union[ir.BoolConst, ir.AND, ir.OR]:
        op = boolops[node.op]
        operands = []
        seen = set()
        for value in node.values:
            value = self.visit(value)
            if value.constant:
                if operator.truth(value):
                    if op == "and":
                        continue
                    else:  # op == "or"
                        return ir.BoolConst(True)
                else:
                    if op == "and":
                        return ir.BoolConst(False)
                    else:  # op == "or"
                        continue
            else:
                if value not in seen:
                    seen.add(value)
                    operands.append(value)
        if len(operands) == 1:
            expr = operands.pop()
            expr = ir.TRUTH(expr)
        else:
            operands = tuple(operands)
            if op == "and":
                expr = ir.AND(operands)
            else:
                expr = ir.OR(operands)
        return expr

    def visit_Compare(self, node: ast.Compare) -> typing.Union[ir.BinOp, ir.BoolOp, ir.BoolConst]:
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        op = compareops[type(node.ops[0])]
        initial_compare = ir.BinOp(left, right, op)
        initial_compare = self.fold_if_constant(initial_compare)
        if len(node.ops) == 1:
            return initial_compare
        compares = [initial_compare]
        seen = set()
        for index, ast_op in enumerate(node.ops[1:], 1):
            left = right
            right = self.visit(node.comparators[index])
            op = compareops[type(ast_op)]
            cmp = ir.BinOp(left, right, op)
            cmp = self.fold_if_constant(cmp)
            if cmp.constant:
                if not operator.truth(cmp):
                    return ir.BoolConst(False)
            else:
                if cmp not in seen:
                    # Preserve the original comparison order, ignoring
                    # duplicate expressions.
                    seen.add(cmp)
                    compares.append(cmp)
        return ir.BoolOp(tuple(compares), "and")

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
            call_ = replace_builtin_call(call_)
        return call_

    def visit_IfExp(self, node: ast.IfExp) -> ir.ValueRef:
        test = self.visit(node.test)
        on_true = self.visit(node.body)
        on_false = self.visit(node.orelse)
        if test.constant:
            if operator.truth(test):
                expr = on_true
            else:
                expr = on_false
        else:
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
        raise TypeError("Extended slices are currently unsupported. This supports single"
                        "slices per dimension")

    def visit_AugAssign(self, node: ast.AugAssign):
        target = self.visit(node.target)
        operand = self.visit(node.value)
        op = binary_inplace_ops.get(type(node.op))
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
                        raise ValueError(msg)
        if node.value is not None:
            # CPython will turn the syntax "var: annotation" into an AnnAssign node
            # with node.value = None. If executed, this won't bind or update the value of var.
            assign = ir.Assign(target, value, pos)
            self.body.append(assign)

    def visit_Pass(self, node: ast.Pass):
        pos = extract_positional_info(node)
        stmt = ir.Pass(pos)
        return stmt

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
            raise ValueError("or else clause not supported for for statements")
        iter_node = self.visit(node.iter)
        target_node = self.visit(node.target)
        pos = extract_positional_info(node)
        targets = []
        iterables = []
        prefix = extract_loop_indexer(node)
        if prefix is None:
            # Use a common prefix if a suitable name was not found.
            prefix = "i"
        loop_index = self.symbols.make_unique_name(prefix)
        # make_loop_interval(targets, iterables, self.symbols, loop_index)

        with self.loop_region(node):
            for target, iterable in unpack_iterated(target_node, iter_node, pos):
                # map each assignment with respect to
                if prefix is None and isinstance(iterable, ir.AffineSeq):
                    if iterable.stop is None:
                        # first enumerate encountered
                        prefix = target
                targets.append(target)
                iterables.append(iterable)
            conflicts = set(targets).intersection(iterables)
            if conflicts:
                conflict_names = ", ".join(c for c in conflicts)
                msg = f"{conflict_names} appear in both the target an iterable sequences of a for loop, " \
                      f"line {pos.line_begin}. This is not supported."
                raise ValueError(msg)
            # make a unique loop index name, which cannot escape the current scope
            if prefix is None:
                prefix = "i"
            loop_index = self.symbols.add_scalar(prefix, self.symbols.default_int, added=True)
            loop_counter = make_loop_counter(iterables, loop_index)

            for stmt in node.body:
                self.visit(stmt)
            # This can still be an ill formed loop, eg one with an unreachable latch.
            # These are not supported, because they look weird after transformation to a legal one
            # and raising an error is less mysterious. They should be be caught during tree validation checks.
            loop = ir.ForLoop(loop_index, loop_counter, self.body, pos)
        self.body.append(loop)

    def visit_While(self, node: ast.While):
        if node.orelse:
            raise ValueError("or else clause not supported for for statements")
        test = self.visit(node.test)
        pos = extract_positional_info(node)
        with self.loop_region(node):
            for stmt in node.body:
                self.visit(stmt)
            loop = ir.WhileLoop(test, self.body, pos)
        self.body.append(loop)

    def visit_Break(self, node: ast.Break):
        if self.enclosing_loop is None:
            raise ValueError("Break encountered outside of loop.")
        pos = extract_positional_info(node)
        stmt = ir.Break(pos)
        self.body.append(stmt)

    def visit_Continue(self, node: ast.Continue):
        if self.enclosing_loop is None:
            raise ValueError("Continue encountered outside of loop.")
        pos = extract_positional_info(node)
        stmt = ir.Continue(pos)
        self.body.append(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # can't check just self.blocks, since we automatically 
        # instantiate entry and exit
        if node is not self.entry:
            raise RuntimeError(f"Nested scopes are unsupported. line: {node.lineno}")
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
        raise NotImplementedError(f"{type(node)} is unsupported")


def map_functions_by_name(node: ast.Module):
    by_name = {}
    assert(isinstance(node, ast.Module))
    for n in node.body:
        if isinstance(n, ast.FunctionDef):
            by_name[n.name] = n
    return by_name


def build_module_ir(src, file_name, type_map):
    """
    Module level point of entry for IR construction.

    Parameters
    ----------

    src: str
        Source code for the corresponding module

    file_name:
        File path we used to extract source. This is used for error reporting.

    type_map:
        Map of function parameters and local variables to numpy or python numeric types.
   
    """

    tree = ast.parse(src, filename=file_name)
    tree = ast.fix_missing_locations(tree)
    import_map = ImportHandler().visit(tree)
    func_asts_by_name = map_functions_by_name(tree)
    funcs = []
    module_symtable = symtable.symtable(src, file_name, "exec")
    build_func_ir = TreeBuilder()
    for func_name, ast_entry_point in func_asts_by_name.items():
        table = module_symtable.lookup(func_name).get_namespace()
        func_type_map = type_map[func_name]
        symbols = symbol_table_from_pysymtable(table, import_map, func_type_map, file_name)
        func_ir = build_func_ir(ast_entry_point, symbols)
        funcs.append(func_ir)

    return ir.Module(funcs, import_map)
