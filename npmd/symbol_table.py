from __future__ import annotations
import builtins
import itertools
import keyword

import numpy as np

from dataclasses import dataclass
from typing import Dict, Optional

import npmd.ir as ir

from npmd.errors import CompilerError
from npmd.utils import extract_name


dtype_pretty_lookup = {
    np.dtype('bool'): 'numpy.bool_',
    np.dtype('uint8'): 'numpy.uint8',
    np.dtype('int8'): 'numpy.int8',
    np.dtype('int32'): 'numpy.int32',
    np.dtype('int64'): 'numpy.int64',
    np.dtype('float32'): 'numpy.float64',
    np.dtype('float64'): 'numpy.float64',
    np.dtype('complex64'): 'numpy.complex64',
    np.dtype('complex128'): 'numpy.complex128'
}


reserved_names = frozenset(set(dir(builtins)).union(set(keyword.kwlist)))


def is_allowed_identifier(name):
    if isinstance(name, ir.NameRef):
        name = name.name
    return isinstance(name, str) and name.isidentifier() and (name not in reserved_names)


def reduces_array_dims(ref):
    if isinstance(ref, ir.NameRef):
        return False
    elif isinstance(ref, ir.Subscript):
        return False if isinstance(ref.index, ir.Slice) else True
    else:
        msg = "{ref} does not represent array view creation."
        raise TypeError(msg)


@dataclass(frozen=True)
class symbol:
    """
    variable name symbol class
    These are meant to be interned by the symbol table and not created arbitrarily.
    """

    def __post_init__(self):
        if self.is_assigned:
            # this should be caught elsewhere, where function information is available
            assert self.is_local
        if self.is_arg:
            if self.type_ is None:
                msg = f'No type declared for argument "{self.name}"'
                raise CompilerError(msg)
            elif not is_allowed_identifier(self.name):
                msg = f'"{self.name}" is not an allowed identifier.'
                raise CompilerError(msg)

    name: str
    type_: Optional[ir.TypeBase]
    is_arg: bool
    is_source_name: bool
    is_local: bool
    is_assigned: bool
    type_is_inferred: bool

    def with_inferred_type(self, t):
        if t is None:
            msg = f'Cannot make typed symbol from "{self.name}" with None.'
            raise CompilerError(msg)
        else:
            if self.type_ is not None:
                # numpy dtypes compare True with None on default for some reason, so this has to be quite specific
                if self.type_ == t:
                    return self
                else:
                    # This is intentionally strict, since inferring too large of an array type is bad.
                    msg = f'Existing type "{self.type_}" is incompatible with updated declaration "{t}"'
                    raise CompilerError(msg)
            return symbol(self.name, t, self.is_arg, self.is_source_name, self.is_local, self.is_assigned, True)


class SymbolTable:
    """
    Per function symbol table with type information and disambiguation of original source vs implementation names.
    """

    def __init__(self, namespace: str, symbols: Dict[str, symbol], default_index_type=np.dtype(np.int_)):
        self.namespace = namespace
        untyped_args = [name for (name, sym) in symbols.items() if sym.is_arg and sym.type_ is None]
        if untyped_args:
            msg = f'Arguments "{untyped_args}" are missing type annotations'
            raise CompilerError(msg)
        self.symbols = symbols
        self.name_manglers = {}
        self.default_index_type = default_index_type

    def drop_symbol(self, name):
        name = extract_name(name)
        del self.symbols[name]
        # keep in name mangler

    @property
    def from_source(self):
        for s in self.symbols.values():
            if s.is_source_name:
                yield ir.NameRef(s.name)

    @property
    def source_locals(self):
        for sym in self.symbols.values():
            if sym.is_source_name and not sym.is_arg:
                yield ir.NameRef(sym.name)

    @property
    def all_locals(self):
        for sym in self.symbols.values():
            if sym.is_local:
                yield ir.NameRef(sym.name)

    @property
    def assigned_names(self):
        for sym in self.symbols.values():
            if sym.is_assigned:
                yield ir.NameRef(sym.name)

    @property
    def arguments(self):
        for sym in self.symbols.values():
            if sym.is_arg:
                yield ir.NameRef(sym.name)

    @property
    def untyped(self):
        for sym in self.symbols.values():
            if sym.type_ is None:
                yield ir.NameRef(sym.name)

    def has_declared_type(self, name: ir.NameRef):
        sym = self.lookup(name)
        return sym.type_ is not None and not sym.type_is_inferred

    def has_inferred_type(self, name: ir.NameRef):
        sym = self.lookup(name)
        return sym.type_is_inferred

    def bind_type(self, name: ir.NameRef, t):
        existing_sym = self.symbols[name.name]
        if existing_sym.type_ is not None and t != existing_sym.type_:
            msg = f'Conflicting types "{existing_sym.type_}" and "{t}" for "{name.name}".'
            raise CompilerError(msg)
        updated_sym = existing_sym.with_inferred_type(t)
        self.symbols[name.name] = updated_sym

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

    def is_assigned(self, name):
        sym = self.lookup(name)
        return sym.is_assigned

    def is_impl_name(self, name):
        sym = self.lookup(name)
        if sym is None:
            return False
        return not sym.is_source_name

    def is_argument(self, name):
        name = extract_name(name)
        sym = self.symbols[name]
        return sym.is_arg

    def check_type(self, name, allow_none=False):
        name = extract_name(name)
        t = self.symbols[name].type_
        if t is None and not allow_none:
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
        if not is_allowed_identifier(prefix_):
            msg = f'Cannot form name from disallowed prefix "{prefix_}"'
            raise CompilerError(msg)
        gen = self._get_name_mangler(prefix_)
        name = f'{prefix_}'
        while self.declares(name):
            name = f'{prefix_}_{next(gen)}'

        sym = symbol(name,
                     type_,
                     is_arg=False,
                     is_source_name=False,
                     is_local=True,
                     is_assigned=True,
                     type_is_inferred=False)

        self.symbols[name] = sym
        # The input name may require mangling for uniqueness.
        # Return the name as it is registered.
        return ir.NameRef(name)

    def make_versioned(self, name):
        assert self.is_source_name(name)
        var_type = self.check_type(name)
        return self.make_unique_name_like(name, var_type)


def dump_symbol_type_info(symbols: SymbolTable):
    print(f'Function: {symbols.namespace} arguments\n')
    for name in symbols.arguments:
        t = symbols.check_type(name)
        if isinstance(t, ir.ArrayType):
            dtype_formatted = dtype_pretty_lookup[t.dtype]
            formatted_type = f'numpy.ndarray[{dtype_formatted}]'
        else:
            formatted_type = dtype_pretty_lookup[t]
        name_to_type = f'{name.name}: {formatted_type}'
        print(name_to_type)
