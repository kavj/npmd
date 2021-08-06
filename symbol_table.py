from __future__ import annotations

import builtins
import itertools
import keyword
import numbers
import operator

import numpy as np

from functools import singledispatch, singledispatchmethod

import ir
import type_resolution as tr

reserved_names = frozenset(set(dir(builtins)).union(set(keyword.kwlist)))


def extract_name(name):
    if not isinstance(name, (str, ir.NameRef)):
        msg = f"Expected a variable name, received type {type(name)}."
        raise TypeError(msg)
    return name.name if isinstance(name, ir.NameRef) else name


def reduces_array_dims(ref):
    if isinstance(ref, ir.NameRef):
        return False
    elif isinstance(ref, ir.Subscript):
        return False if isinstance(ref.slice, ir.Slice) else True
    else:
        msg = "{ref} does not represent array view creation."
        raise TypeError(msg)


@singledispatch
def wrap_input(input):
    msg = f"No method to wrap {input} of type {type(input)}."
    raise NotImplementedError(msg)


@wrap_input.register
def _(input: str):
    if not input.isidentifier():
        msg = f"{input} is not a valid variable name."
        raise ValueError(msg)
    return ir.NameRef(input)


@wrap_input.register
def _(input: ir.NameRef):
    return input


@wrap_input.register
def _(input: ir.Constant):
    return input


@wrap_input.register
def _(input: int):
    return ir.IntConst(input)


@wrap_input.register
def _(input: bool):
    return ir.BoolConst(input)


@wrap_input.register
def _(input: numbers.Integral):
    return ir.IntConst(input)


@wrap_input.register
def _(input: numbers.Real):
    return ir.FloatConst(input)


class FunctionContext:

    def __init__(self, func, types):
        self.func = func
        self.types = types


class CompilerContext:

    def __init__(self, funcs, type_maps):
        self.funcs = funcs
        self.type_maps = type_maps


def map_alias_to_qualified_names(import_nodes):
    """
    Internally, we refer to qualified names for uniqueness reasons.
    This maps any any aliases of modules or names from modules to
    qualified names.

    alias: module_name or alias: module_name.imported_name

    """
    qual_names = {}
    for node in import_nodes:
        if isinstance(node, ir.NameImport):
            qual_names[node.as_name] = f"{node.module}.{node.name}"
        elif isinstance(node, ir.ModImport):
            qual_names[node.as_name] = node.module
        else:
            raise ValueError


class symbol:
    """
    variable name symbol class
    """

    def __init__(self, name, type_, make_unique):
        self.name = name
        self.type_ = type_
        self.added = make_unique

    def __eq__(self, other):
        assert isinstance(other, symbol)
        return (self.name == other.name
                and self.type_ == other.type_
                and self.added == other.added)

    def __ne__(self, other):
        assert isinstance(other, symbol)
        return (self.name != other.name
                or self.type_ != other.type_
                or self.added != other.added)

    def __hash__(self):
        return hash(self.name)

    @property
    def is_array(self):
        return isinstance(self.type_, (ir.ArrayRef, ir.ViewRef))

    @property
    def is_integer(self):
        return self.type_ in (tr.Int32, tr.Int64)


# array creation nodes

def make_numpy_call(node: ir.Call):
    name = node.func
    if name == "numpy.ones":
        fill_value = ir.One
    elif name == "numpy.zeros":
        fill_value = ir.Zero
    else:
        if name != "numpy.empty":
            raise NotImplementedError
        fill_value = None
    args = node.args
    kwargs = node.keywords
    if not (1 <= len(args) + len(kwargs) <= 2):
        raise ValueError
    params = {}
    for name, value in zip(("shape", "dtype"), args):
        params[name] = value
    for key, value in kwargs:
        if key in params:
            raise KeyError
        params[key] = value
    shape = params["shape"]
    dtype = params.get("dtype", np.float64)
    # Todo: initializer didn't match the rest of this module. Rewrite later.
    array_init = ()
    return array_init


class symboltable:
    type_by_name = {"int": int, "float": float, "numpy.int32": np.int32, "numpy.int64": np.int64,
                    "numpy.float32": np.float32, "numpy.float64": np.float64}

    scalar_ir_types = frozenset({tr.Int32, tr.Int64, tr.Float32, tr.Float64, tr.Predicate32, tr.Predicate64})

    def __init__(self, scope_name, src_locals, import_map):
        self.symbols = {}   # name -> typed symbol entry
        self.scalar_type_map = tr.scalar_type_map.copy()
        self.src_locals = src_locals # initially declared names
        self.import_map = import_map # source name -> qualified name
        self.scope_name = scope_name
        self.prefixes = {}  # prefix for adding enumerated variable names

    def make_type_lowering_rule(self, input_type, lltype):
        assert isinstance(input_type, type)
        assert lltype in symboltable.scalar_ir_types
        self.scalar_type_map[input_type] = lltype

    def declares(self, name):
        name = wrap_input(name)
        return name in self.symbols

    def get_ir_type(self, type_):
        if isinstance(type_, ir.NameRef):
            type_ = symboltable.type_by_name.get(type_.name)
            if type_ is None:
                msg = f"Cannot match ir type to {type}."
                raise ValueError(msg)
        if isinstance(type_, ir.ArrayType):
            dtype = type_.dtype
            if dtype not in symboltable.scalar_ir_types:
                dtype = self.scalar_type_map.get(dtype)
                if dtype is None:
                    msg = f"Cannot match ir type to array element type {type_.dtype}."
                    raise ValueError(msg)
            type_ = ir.ArrayType(type_.ndims, dtype)
        elif type_ not in symboltable.scalar_ir_types:
            dtype = self.scalar_type_map.get(type_)
            if dtype is None:
                msg = f"Cannot match ir type to scalar type {type_}"
                raise ValueError(msg)
            type_ = dtype
        return type_

    def lookup(self, name):
        name = wrap_input(name)
        sym = self.symbols.get(name)
        return sym

    def is_added(self, name):
        name = wrap_input(name)
        return name not in self.src_locals

    @property
    def default_int(self):
        return self.scalar_type_map[int]

    def _get_num_generator(self, prefix):
        # splitting by prefix helps avoids appending
        # large numbers in most cases
        gen = self.prefixes.get(prefix)
        if gen is None:
            gen = itertools.count()
            self.prefixes[prefix] = gen
        return gen

    def is_array(self, name):
        sym = self.lookup(name)
        return isinstance(sym.type_, ir.ArrayType)

    def make_symbol(self, name, type_, added):
        name = wrap_input(name)
        if self.declares(name):
            existing = self.lookup(name)
            msg = f"Existing symbol definition {existing} for {name} is incompatible with new definition" \
                  f"{type_}."
            raise KeyError(msg)
        else:
            sym = symbol(name, type_, added)
            self.symbols[name] = sym

    def make_unique_name(self, prefix):
        prefix = extract_name(prefix)
        if self.declares(prefix):
            gen = self._get_num_generator(prefix)
            name = f"{prefix}_{next(gen)}"
            while self.declares(name):
                name = f"{prefix}_{next(gen)}"
            name = ir.NameRef(name)
        else:
            # No need to rename
            name = ir.NameRef(prefix)
        return name

    # These return a name reference to the symbol name that is actually used.
    # In the case of is_added=False, this is the original name. Otherwise it
    # may be a unique variable name that instantiates some renaming or implementation binding.

    def add_typed_scalar(self, name, type_, added):
        # run type check by default, to avoid internal array types
        # with Python or numpy types as parameters.

        name = wrap_input(name)
        if added:
            name = self.make_unique_name(name)
        if self.declares(name):
            msg = f"Duplicate symbol declaration for variable name '{name}'."
            raise ValueError(msg)
        type_ = self.get_ir_type(type_)
        self.make_symbol(name, type_, added)
        return name

    @singledispatchmethod
    def add_view(self, base, name, transpose, is_added):
        raise NotImplementedError

    @add_view.register
    def _(self, expr: ir.Subscript, name, is_added):
        base = self.lookup(expr.value)
        base_type = base.type_
        if not isinstance(base_type, ir.ArrayType):
            msg = f"Cannot take view of non-array type {base.name}"
            raise ValueError(msg)
        if is_added or name not in self.src_locals:
            # mangle if either the name or assignment motivating
            # this symbol is not present in source.
            name = self.make_unique_name(prefix=name)
        if isinstance(expr.slice, ir.Slice):
            # no dim change
            ndims = base_type.ndims
        else:
            ndims = operator.sub(base_type.ndims, 1)
        if operator.ge(ndims.value, 1):
            view_type = ir.ArrayType(ndims, base_type.dtype)
        else:
            view_type = base_type.dtype
        self.make_symbol(name, view_type, is_added)
        return name

    @add_view.register
    def _(self, expr: ir.NameRef, name, is_added):
        base = self.lookup(expr)
        base_type = base.type_
        if not isinstance(base_type, ir.ArrayType):
            msg = f"Cannot take view using non-array type {base_type}"
            raise ValueError(msg)
        if is_added or name not in self.src_locals:
            name = self.make_unique_name(name)
        self.make_symbol(name, base_type, is_added)
        return name


def symbol_table_from_pysymtable(func_table, import_map, type_map, file_name):
    """
    Build an internally used symbol table from a Python function symtable and type information.

    func_table: Function symbol table symtable.symtable
    type_map: dict[name: interface_type]
    file_name: source file name

    """
    func_name = func_table.get_name()
    if func_table.is_nested():
        raise ValueError(f"{func_name} in file {file_name} appears as a nested scope, which is unsupported.")
    elif func_table.has_children():
        raise ValueError(f"{func_name} in file {file_name} contains nested scopes, which are unsupported.")
    elif func_table.get_type() != "function":
        raise TypeError(f"{func_name} in file {file_name} refers to a class rather than a function. This is "
                        f"unsupported.")
    missing = []
    locals_from_source = set(func_table.get_locals())
    table = symboltable(func_name, locals_from_source, import_map)

    # register types
    for arg in func_table.get_parameters():
        # Check that all arguments have type info.
        type_ = type_map.get(arg)
        table.add_typed_scalar(arg, type_, added=False)
    s = table.symbols.get(ir.NameRef('a'))
    if missing:
        args = ", ".join(arg for arg in missing)
        msg = f"Function '{func_table.get_name()}' is missing type info for the following arguments: {args}."
        raise ValueError(msg)
    locals_from_source = set(func_table.get_locals())
    if func_table.get_name() in locals_from_source:
        msg = f"Function {func_table.get_name()} contains a local variable with the same name. This is unsupported."
        raise ValueError(msg)

    return table
