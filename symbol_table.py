import itertools

from contextlib import contextmanager
from symtable import symtable, Function, Symbol

import ir
import type_resolution as tr

from errors import CompilerError
from utils import extract_name, wrap_input


def reduces_array_dims(ref):
    if isinstance(ref, ir.NameRef):
        return False
    elif isinstance(ref, ir.Subscript):
        return False if isinstance(ref.slice, ir.Slice) else True
    else:
        msg = "{ref} does not represent array view creation."
        raise TypeError(msg)


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
    These are meant to be interned by the symbol table and not created arbitrarily.
    """

    def __init__(self, name: str, is_source_name, is_arg, is_assigned):
        self.name = name
        self.is_source_name = is_source_name
        self.is_arg = is_arg
        self.is_assigned = is_assigned

    def __eq__(self, other):
        assert isinstance(other, symbol)
        return (self.name == other.name
                and self.is_source_name == other.is_source_name)

    def __ne__(self, other):
        assert isinstance(other, symbol)
        return (self.name != other.name
                or self.is_source_name != other.is_source_name)

    def __hash__(self):
        return hash(self.name)


class func_symbol_table:
    scalar_ir_types = frozenset({tr.Int32, tr.Int64, tr.Float32, tr.Float64, tr.Predicate32, tr.Predicate64})

    def __init__(self, func: Function, default_int):
        self.symbols = {}  # name -> typed symbol entry
        self.types = {}
        self.default_int = default_int
        name = func.get_name()
        self.func_name = name
        if func.is_nested():
            raise ValueError(f"{name} in file {file_name} appears as a nested scope, which is unsupported.")
        elif func.has_children():
            raise ValueError(f"{name} in file {file_name} contains nested scopes, which are unsupported.")
        elif func.get_type() != "function":
            raise TypeError(f"{name} in file {file_name} refers to a class rather than a function. This is "
                            f"unsupported.")
        for s in func.get_symbols():
            if s.is_imported():
                msg = f"No current support for function scope imports."
                raise CompilerError(msg)
            sym = symbol(name=s.get_name(), is_source_name=True, is_arg=s.is_parameter(), is_assigned=s.is_assigned())
            self.symbols[sym.name] = sym
        self.prefixes = {}  # prefix for adding enumerated variable names

    def declares(self, name):
        return name in self.symbols

    def lookup(self, name):
        sym = self.symbols.get(name)
        return sym

    def is_source_name(self, name):
        sym = self.lookup(name)
        if sym is None:
            return False
        return sym.is_source_name

    def check_type(self, name):
        return self.types.get(name)

    def get_arguments(self):
        return {s for s in self.symbols if s.is_arg}

    def _get_num_generator(self, prefix: str):
        # splitting by prefix helps avoids appending
        # large numbers in most cases
        gen = self.prefixes.get(prefix)
        if gen is None:
            gen = itertools.count()
            self.prefixes[prefix] = gen
        return gen

    def make_unique_name_like(self, name, type_ = None):
        """
        This is used to add a unique typed temporary variable name.
        """

        name = extract_name(name)
        if type_ is None:
            # If we're value numbering a name, this grabs type info from the base name.
            type_ = self.check_type(name)
            if type_ is None:
                msg = f"Failed to retrieve a type for name {name}."
                raise CompilerError(msg)
        gen = self._get_num_generator(prefix_)
        while self.declares(name):
            name = wrap_input(f"{prefix_}_{next(gen)}")
        name = self.make_unique_name(name)
        sym = symbol(name, is_source_name=False, is_arg=False, is_assigned=True)
        self.symbols[name] = sym
        self.declare_type(name, type_)
        # The input name may require mangling for uniqueness.
        # Return the name as it is registered.
        return name


class module_symbol_table:
    def __init__(self, name, default_int_is_64=True):
        self.name = name
        self.default_int = tr.Int64 if default_int_is_64 else tr.Int32
        self.funcs = {}
        self.func_imports = {}
        self.mod_imports = {}

    # Todo: Name import must verify that it only imports a method defined in some other module

    def lookup_func(self, name):
        value = self.funcs.get(name)
        if value is None:
            msg = f"No function {name} is known for module {self.name}."
            raise CompilerError(msg)
        return value

    def register_name_import(importref):
        if not isinstance(importref, ir.NameImport):
            raise TypeError(f"{importref} is not a name import")
        name = importref.name
        if name in self.func_imports or name in self.mod_imports:
            raise CompilerError(f"Duplicate im")

    def register_module_import(importref):
        # if not isinstance(importref, ir.ModImpport for name {name}.")
        # self.func_imports[name] = importrefort):
        #    raise TypeError(f"{importref} is not a module import")
        name = importref.as_name
        if name in self.func_imports or name in self.mod_imports:
            raise CompilerError(f"Duplicate import for name {name}.")

    def register_func(self, func: func_symbol_table):
        name = func.func_name
        if name in self.funcs:
            msg = f"Function {name} shadows an existing symbol in module {self.name}"
            raise CompilerError(msg)
        self.funcs[name] = func


def build_module_symbol_table(src, name):
    module = module_symbol_table(name)
    top = symtable(src, name, "exec")
    # use default int == 64 for now. This could be made platform specific
    # and overridable here
    for func in top.get_children():
        name = func.get_name()
        if func.is_nested():
            raise ValueError(f"{name} in file {file_name} appears as a nested scope, which is unsupported.")
        elif func.has_children():
            raise ValueError(f"{name} in file {file_name} contains nested scopes, which are unsupported.")
        elif func.get_type() != "function":
            raise TypeError(f"{name} in file {file_name} refers to a class rather than a function. This is "
                            f"unsupported.")
        func_table = func_symbol_table(func, tr.Int64)
        module.register_func(func_table)
    return module
