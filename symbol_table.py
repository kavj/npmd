import itertools

from dataclasses import dataclass
from typing import Dict

import ir

from errors import CompilerError
from utils import extract_name, is_allowed_identifier


def reduces_array_dims(ref):
    if isinstance(ref, ir.NameRef):
        return False
    elif isinstance(ref, ir.Subscript):
        return False if isinstance(ref.index, ir.Slice) else True
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


@dataclass(frozen=True)
class symbol:
    """
    variable name symbol class
    These are meant to be interned by the symbol table and not created arbitrarily.
    """

    name: str
    type_: ir.TypeBase
    is_arg: bool
    is_source_name: bool
    is_local: bool


class SymbolTable:
    """
    Per function symbol table with type information and disambiguation of original source vs implementation names.
    """

    def __init__(self, namespace: str, symbols: Dict[str, symbol]):
        self.namespace = namespace
        self.symbols = symbols
        self.name_manglers = {}

    @property
    def from_source(self):
        for s in self.symbols.values():
            if s.is_source_name:
                yield s

    @property
    def source_locals(self):
        for sym in self.symbols.values():
            if sym.is_source_name and not sym.is_arg:
                yield sym

    @property
    def all_locals(self):
        for sym in self.symbols.values():
            if sym.is_local:
                yield sym

    @property
    def arguments(self):
        for sym in self.symbols.values():
            if sym.is_arg:
                yield sym

    def declares(self, name):
        name = extract_name(name)
        return name in self.symbols

    def lookup(self, name):
        name = extract_name(name)
        sym = self.symbols.get(name)
        return sym

    def is_source_name(self, name):
        sym = self.lookup(name)
        return (sym is not None
                and sym.is_source_name)

    def is_impl_name(self, name):
        sym = self.lookup(name)
        if sym is None:
            return False
        return not sym.is_source_name

    def check_type(self, name):
        name = extract_name(name)
        t = self.symbols[name].type_
        if t is None:
            msg = f"No type declared for symbol '{name}' in namespace '{str(self.namespace)}'."
            raise CompilerError(msg)
        return t

    def check_dtype(self, name):
        base_type = self.check_type(name)
        if isinstance(base_type, ir.ArrayType):
            base_type = base_type.dtype
        return base_type

    def _get_name_mangler(self, prefix: str):
        # splitting by prefix helps avoids appending
        # large numbers in most cases
        gen = self.name_manglers.get(prefix)
        if gen is None:
            gen = itertools.count()
            self.name_manglers[prefix] = gen
        return gen

    def make_unique_name_like(self, name, type_):
        """
        This is used to add a unique typed temporary variable name.
        """
        prefix_ = extract_name(name)
        if type_ is None:
            msg = f'Failed to retrieve a type for name {prefix_}.'
            raise CompilerError(msg)
        elif not is_allowed_identifier(prefix_):
            msg = f'Cannot form name from disallowed prefix "{prefix_}"'
            raise CompilerError(msg)
        gen = self._get_name_mangler(prefix_)
        name = f'{prefix_}'
        while self.declares(name):
            name = f'{prefix_}_{next(gen)}'
        sym = symbol(name, type_, is_arg=False, is_source_name=False, is_local=True)
        self.symbols[name] = sym
        # The input name may require mangling for uniqueness.
        # Return the name as it is registered.
        return ir.NameRef(name)

    def make_alias(self, name):
        var_type = self.check_type(name)
        return self.make_unique_name_like(name, var_type)
