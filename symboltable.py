import builtins
import itertools
import keyword

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
        self.gen = itertools.count()

    def __contains__(self, item):
        if isinstance(item, ir.NameRef):
            item = item.name
        return item in self.names

    def make_unique_name(self, prefix):
        name = f"{prefix}_{next(self.gen)}"
        while name in self.names:
            name = f"{prefix}_{next(self.gen)}"
        self.names.add(name)
        return name


def create_symbol_tables(src, filename):
    tables = {}
    mod = symtable(src, filename, "exec")
    for func in mod.get_children():
        if func.get_type() == "class":
            raise TypeError(f"Classes are not supported.")
        elif func.has_children():
            raise ValueError(f"Nested scopes are not supported")
        var_names = set()
        # we'll eventually need back end reserved names
        # but probably not here
        for identifier in func.get_symbols():
            name = identifier.get_name()
            if name in reserved_names:
                if identifier.is_assigned():
                    raise NotImplementedError(f"Reassigning names used by the language itself is unsupported. "
                                              "{name} marked as assignment target")
            else:
                var_names.add(name)
        funcname = func.get_name()
        tables[funcname] = var_names
    return tables
