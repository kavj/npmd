import builtins
import itertools
import keyword

from collections import defaultdict
from symtable import symtable

import ir

reserved_names = frozenset(set(dir(builtins)).union(set(keyword.kwlist)))


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


def canonicalize_type_map(by_type, canonical_types):
    """
    Map type parameterized name sets to use canonical types.
    This is intended to resolve type conflicts where two types
    map to the same low level canonical type.

    """
    repl = defaultdict(set)
    for t, names in by_type.items():
        ct = canonical_types.get(t)
        if ct is None:
            msg = f"Cannot map unsupported type {t}"
            raise TypeError(msg)
        repl[ct].update(names)
    return repl


def bind_type_by_name(types):
    """
    in:
        types: dict[type: set(all variable names of this type...)]

    out:
        by_name: dict[name: type]

    """
    by_name = {}
    for type_, names in types.items():
        for name in names:
            if name in by_name:
                first = by_name[name]
                msg = f"Duplicate type entry {first} and {type_} for name {name}"
                raise ValueError(msg)
            by_name[name] = type_


def create_symbol_tables(src, filename, types):
    tables = {}
    mod = symtable(src, filename, "exec")
    for func in mod.get_children():
        if func.get_type() == "class":
            raise TypeError(f"Classes are not supported.")
        elif func.has_children():
            raise ValueError(f"Nested scopes are not supported")
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
        funcname = func.get_name()
        table = symbol_gen(var_names)
        tables[funcname] = table
    return tables
