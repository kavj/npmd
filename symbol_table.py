from __future__ import annotations

import builtins
import itertools
import keyword
import numbers

import numpy as np

from functools import singledispatch, singledispatchmethod

import ir
import type_resolution as tr

from errors import CompilerError

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


def is_valid_identifier(input):
    if isinstance(input, ir.NameRef):
        input = input.name
    return isinstance(input, str) and input.isidentifier() and (input not in reserved_names)


@singledispatch
def wrap_input(input):
    msg = f"No method to wrap {input} of type {type(input)}."
    raise NotImplementedError(msg)


@wrap_input.register
def _(input: str):
    if not is_valid_identifier(input):
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

    def __init__(self, name, type_, is_source_name, is_arg, is_assigned):
        assert isinstance(name, ir.NameRef)
        assert is_source_name or type_ is not None
        self.name = name
        self.type_ = type_
        self.is_source_name = is_source_name
        self.is_arg = is_arg
        self.is_assigned = is_assigned

    def __eq__(self, other):
        assert isinstance(other, symbol)
        return (self.name == other.name
                and self.type_ == other.type_
                and self.is_source_name == other.is_source_name)

    def __ne__(self, other):
        assert isinstance(other, symbol)
        return (self.name != other.name
                or self.type_ != other.type_
                or self.is_source_name != other.is_source_name)

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
    scalar_ir_types = frozenset({tr.Int32, tr.Int64, tr.Float32, tr.Float64, tr.Predicate32, tr.Predicate64})

    def __init__(self, scope_name, default_int_is_64=True):
        self.symbols = {}   # name -> typed symbol entry
        self.default_int = tr.Int64 if default_int_is_64 else tr.Int32
        self.scope_name = wrap_input(scope_name)  # wrap everything to avoid false comparing raw strings to namerefs
        self.prefixes = {}  # prefix for adding enumerated variable names

    def declares(self, name):
        name = wrap_input(name)
        return name in self.symbols

    def lookup(self, name):
        name = wrap_input(name)
        sym = self.symbols.get(name)
        return sym

    def is_source_name(self, name):
        sym = self.lookup(name)
        if sym is None:
            return False
        return sym.is_source_name

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

    def lookup_type(self, name):
        """
        shortcut to get type info only
        """
        sym = self.lookup(name)
        return sym.type_

    def is_typed(self, name):
        sym = self.lookup(name)
        return sym is not None and sym.type_ is not None

    def make_unique_name(self, prefix):
        prefix = extract_name(prefix)
        if self.declares(prefix):
            gen = self._get_num_generator(prefix)
            name = wrap_input(f"{prefix}_{next(gen)}")
            while name in self.symbols:
                name = wrap_input(f"{prefix}_{next(gen)}")
        else:
            # First use, no need to rename
            name = wrap_input(prefix)
        return name

    def bind_type_to_name(self, name, type_):
        """
        Register a type for an untyped or undeclared symbol.
        """

        assert type_ is not None
        sym = self.lookup(name)
        if symbol is None:
            msg = f"Internal Error: Cannot add type to undeclared symbol name {name}."
            raise CompilerError(msg)
        else:
            declared_type = sym.type_
            if declared_type is None:
                sym.type_ = type_
            elif declared_type != type_:
                msg = f"Conflicting types {declared_type} and {type_} for variable name {name}."
                raise CompilerError(msg)

    def register_src_name(self, name, is_arg, is_assigned, type_):
        """
        Register a source name using the corresponding symbol from the Python symbol table.

        """

        if self.declares(name):
            msg = f"Internal Error: Source name {name} is already registered."
            raise KeyError(msg)
        name = wrap_input(name)
        if name == self.scope_name:
            msg = f"Variable name: {name.name} shadows function name."
            raise CompilerError(msg)
        is_source_name = True
        sym = symbol(name, type_, is_source_name, is_arg, is_assigned)
        self.symbols[name] = sym

    def register_impl_name(self, name, type_):
        """
        This is used to add a unique typed temporary variable name.
        """
        name = self.make_unique_name(name)
        sym = symbol(name, type_, is_source_name=False, is_arg=False, is_assigned=True)
        self.symbols[name] = sym
        # The input name may require mangling for uniqueness.
        # Return the name as it is registered.
        return name


def symbol_table_from_pysymtable(func_table, type_map, file_name):
    """
    Build an internally used symbol table from a Python function symtable and type information.
    Note: read func_table as py_symbol_table, noting that calling it py_anything would be a Python language violation.

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
    internal_table = symboltable(func_name)

    # register types
    for name in func_table.get_locals():
        sym = func_table.lookup(name)
        if sym.is_imported():
            msg = f"Imports at function scope are not currently supported. Import alias: {name}"
            raise CompilerError(msg)
        is_arg = sym.is_parameter()
        is_assigned = sym.is_assigned()
        type_ = None
        internal_table.register_src_name(name, is_arg, is_assigned, type_)

    return internal_table
