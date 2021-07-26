from __future__ import annotations

import builtins
import itertools
import keyword
import numbers
import operator

import numpy as np

from functools import singledispatch, singledispatchmethod
from symtable import symtable

import ir


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
    return ir.IntNode(input)


@wrap_input.register
def _(input: bool):
    return ir.BoolNode(input)


@wrap_input.register
def _(input: numbers.Integral):
    return ir.IntNode(input)


@wrap_input.register
def _(input: numbers.Real):
    return ir.FloatNode(input)


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
            qual_names[node.asname] = f"{node.mod}.{node.name}"
        elif isinstance(node, ir.ModImport):
            qual_names[node.asname] = node.mod
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
        return isinstance(self.type_, (ir.Int32, ir.Int64))


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

    scalar_types = frozenset({ir.Int32, ir.Int64, ir.Float32, ir.Float64, ir.Predicate32, ir.Predicate64})

    _scalar_lookup = {int: ir.Int64, float: ir.Float64, np.int32: ir.Int32, np.int64: ir.Int64, np.float32: ir.Float32,
                      np.float64: ir.Float64}

    def __init__(self, scope_name, src_locals, default_int_type=ir.Int64):
        self.symbols = {}
        self.scalar_type_map = symboltable._scalar_lookup.copy()
        self.scalar_type_map[int] = default_int_type
        self.src_locals = src_locals
        self.scope_name = scope_name
        self.prefixes = {}  # prefix for adding enumerated variable names

    def declares(self, name):
        name = wrap_input(name)
        return name in self.symbols

    def get_internal_type(self, type_):
        if isinstance(type_, ir.ArrayType):
            dtype = type_.dtype
            if dtype not in symboltable.scalar_types:
                dtype = self.scalar_type_map.get(dtype)
                if dtype is None:
                    msg = f"Cannot map {type_.dtype} to an internal scalar type."
                    raise TypeError(msg)
                type_ = ir.ArrayType(type_.ndims, dtype)
        elif type_ not in symboltable.scalar_types:
            dtype = self.scalar_type_map.get(type_)
            if dtype is None:
                msg = f"Cannot map {type_} to an internal scalar type."
                raise TypeError(msg)
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

    def add_scalar(self, name, type_, added):
        # run type check by default, to avoid internal array types
        # with Python or numpy types as parameters.

        name = wrap_input(name)
        if added:
            name = self.make_unique_name(name)
        if self.declares(name):
            msg = f"Duplicate symbol declaration for variabl name '{name}'."
            raise ValueError(msg)
        type_ = self.get_internal_type(type_)
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


def make_internal_symbol_table(func_table, type_map, file_name):
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
    for arg in func_table.get_parameters():
        # Check that all arguments have type info.
        if arg not in type_map:
            missing.append(arg)
    if missing:
        args = ", ".join(arg for arg in missing)
        msg = f"Function '{func_table.get_name()}' is missing type info for the following arguments: {args}."
        raise ValueError(msg)
    # We track these explicitly for a couple reasons. First, names defined by the language but overwritten
    # here will be handled incorrectly if we don't make note of them here. Second, variables added for instrumentation
    # have enforced single assignments and tighter declaration bounds.
    locals_from_source = set(func_table.get_locals())
    # This might be handled by renaming later. We throw an error here, because it's
    # almost certainly unintentional, and this is intended to be compiled ahead of execution
    # rather than milliseconds prior to execution.
    if func_table.get_name() in locals_from_source:
        msg = f"Function {func_table.get_name()} contains a local variable with the same name. This is unsupported."
        raise ValueError(msg)
    # Todo: extract source names

    # table = symboltable(type_builder)
    # for name, type_ in type_map.items():
    #    table.add_var(name, type_, added=False)
    # return table


def create_symbol_tables(src, filename, types_by_func, use_default_int64=True):
    tables = {}
    mod = symtable(src, filename, "exec")
    # extract names that correspond to functions
    for func in mod.get_children():
        name = func.get_name()
        types = types_by_func.get(name, ())
        from_src = set(func.get_locals())
        tables[name] = make_internal_symbol_table(func, types, filename)
    return tables
