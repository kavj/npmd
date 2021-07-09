import builtins
import inspect
import importlib
import itertools
import keyword

import numpy as np

from collections import defaultdict
from symtable import symtable

import ir

reserved_names = frozenset(set(dir(builtins)).union(set(keyword.kwlist)))


class ArrayCreationInitializer:
    def __init__(self, dims, dtype, fill_value):
        self.dims = dims
        self.dtype = dtype
        self.fill_value = fill_value


class TypeBuilder:

    def __init__(self, default_int64=True):
        int32_type = ir.ScalarType(signed=True, boolean=False, integral=True, bitwidth=32)
        int64_type = ir.ScalarType(signed=True, boolean=False, integral=True, bitwidth=64)
        float32_type = ir.ScalarType(signed=True, boolean=False, integral=False, bitwidth=32)
        float64_type = ir.ScalarType(signed=True, boolean=False, integral=False, bitwidth=64)
        bool_type = ir.ScalarType(signed=True, boolean=True, integral=True, bitwidth=1)
        types = {np.int32: int32_type, np.int64: int64_type, float32_type: np.float32, float64_type: np.float64,
                 bool: bool_type}
        if default_int64:
            types[int] = int64_type
        else:
            types[int] = int32_type
        # Python floats are always double precision
        types[float] = float64_type
        self.types = types

    @property
    def default_float(self):
        return self.types[float]

    @property
    def default_int(self):
        return self.types[int]


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

# array creation nodes


def make_numpy_call(node: ir.Call):
    name = node.funcname
    if name == "numpy.ones":
        fill_value = ir.IntNode(1)
    elif name == "numpy.zeros":
        fill_value = ir.IntNode(0)
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
    array_init = ArrayCreationInitializer(shape, dtype, fill_value)
    return array_init


def extract_name(name):
    return name.name if isinstance(name, ir.NameRef) else name


class name_generator:
    def __init__(self, prefix):
        self.prefix = prefix
        self.gen = itertools.count()

    def make_name(self):
        return f"{self.prefix}_{next(self.gen)}"


class symbol_gen:
    def __init__(self, existing):
        self.names = existing
        self.added = set()
        self.prefixes = {}
        self.arrays = {}

    def __contains__(self, item):
        if isinstance(item, ir.NameRef):
            item = item.name
        return item in self.names

    def _get_num_generator(self, prefix):
        # splitting by prefix helps avoids appending
        # large numbers in most cases
        gen = self.prefixes.get(prefix)
        if gen is None:
            gen = itertools.count()
            self.prefixes[prefix] = gen
        return gen

    def is_array(self, name):
        return name in self.arrays

    def add_array_view(self, name, base, subscript):
        if name in self.arrays:
            # Aliasing and multiple parameter combinations become too tedious
            # if we have multiple possible assignments to a given array.
            raise KeyError
        view = ir.ViewRef(base, subscript)
        self.arrays[name] = view

    def add_array(self, name, dims, elem_type):
        if name in self.arrays:
            # Aliasing and multiple parameter combinations become too tedious
            # if we have multiple possible assignments to a given array.
            raise KeyError
        arr = ir.ArrayRef(dims, elem_type)
        self.arrays[name] = arr

    def make_unique_name(self, prefix):
        gen = self._get_num_generator(prefix)
        name = f"{prefix}_{next(gen)}"
        while name in self.names:
            name = f"{prefix}_{next(gen)}"
        self.names.add(name)
        self.added.add(name)
        return name


def unify_types(by_type, canonical_types):
    """
    Map type parameterized name sets to use canonical types.
    This may merge type aliases that otherwise appear incompatible.

    """
    repl = defaultdict(set)
    for t, names in by_type.items():
        ct = canonical_types.get(t)
        if ct is None:
            msg = f"Cannot map unsupported type {t}"
            raise TypeError(msg)
        repl[ct].update(names)
    return repl


def bind_types_to_names(types):
    """
    Convert from a dictionary mapping types to sets of variable names
    to a dictionary mapping each distinct variable name to a single type.

    This should run after type unification to avoid false conflicts.

    """
    by_name = {}
    for type_, names in types.items():
        for name in names:
            if name in by_name:
                first = by_name[name]
                msg = f"Duplicate type entry {first} and {type_} for name {name}"
                raise ValueError(msg)
            by_name[name] = type_


def create_symbol_tables(src, filename, types_by_func):
    tables = {}
    mod = symtable(src, filename, "exec")
    # extract names that correspond to functions
    for func in mod.get_children():
        func_name = func.get_name()
        if func.is_nested():
            raise ValueError(f"{func_name} in file {filename} appears as a nested scope, which is unsupported.")
        elif func.has_children():
            raise ValueError(f"{func_name} in file {filename} contains nested scopes, which are unsupported.")
        elif func.get_type() != "function":
            raise TypeError(f"{func_name} in file {filename} refers to a class rather than a function. This is "
                            f"unsupported.")
        if func_name not in types_by_func:
            raise ValueError(f"No type information provided for function {func_name}")
        var_names = set()
        args = set()
        # we'll eventually need back end reserved names
        # but probably not here
        for sym in func.get_symbols():
            name = sym.get_name()
            if name in reserved_names:
                if sym.is_assigned():
                    raise NotImplementedError(f"Reassigning names used by the language itself is unsupported. "
                                              "{name} marked as assignment target")
            else:
                var_names.add(name)
                if sym.is_parameter():
                    args.add(name)
        table = symbol_gen(var_names)
        tables[func_name] = table
    return tables
