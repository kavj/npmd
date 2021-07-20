import builtins
import itertools
import keyword
import numbers

import numpy as np

from functools import singledispatch
from symtable import symtable

import ir

from TypeInterface import ArrayInput, ArrayView, ScalarType

reserved_names = frozenset(set(dir(builtins)).union(set(keyword.kwlist)))


class ArrayCreationInitializer:
    def __init__(self, dims, dtype, fill_value):
        self.dims = dims
        self.dtype = dtype
        self.fill_value = fill_value


class TypeLookup:
    def __init__(self):
        # partly intern basic types
        # Predicates don't really differ between integral and floating point representations,
        # but AVX intrinsics distinguish between the two, based on the operand types that generated
        # the predicate mask and how it is used.
        self.types = {"int32": ScalarType(bitwidth=32, integral=True, boolean=False),
                      "int64": ScalarType(bitwidth=64, integral=True, boolean=False),
                      "float32": ScalarType(bitwidth=32, integral=False, boolean=False),
                      "float64": ScalarType(bitwidth=64, integral=False, boolean=False),
                      "pred32": ScalarType(bitwidth=32, integral=True, boolean=True),
                      "pred64": ScalarType(bitwidth=64, integral=True, boolean=True),
                      "fpred32": ScalarType(bitwidth=32, integral=False, boolean=True),
                      "fpred64": ScalarType(bitwidth=64, integral=False, boolean=True)}

    def lookup(self, type_):
        t = self.types.get(type_)
        if t is None:
            msg = f"No internal type matches name type name {type_}."
            raise KeyError(msg)
        return t


@singledispatch
def wrap_array_parameter(param):
    if param is not None:
        msg = f"{param} cannot be coerced to a suitable array parameter type."
        raise TypeError(msg)


@wrap_array_parameter.register
def _(param: str):
    return ir.NameRef(param)


@wrap_array_parameter.register
def _(param: numbers.Integral):
    # numbers.Integral catches numpy integral types
    # which are not caught by int
    return ir.IntNode(param)


class TypeBuilder:
    # partly intern basic types
    # Predicates don't really differ between integral and floating point representations,
    # but AVX intrinsics distinguish between the two, based on the operand types that generated
    # the predicate mask and how it is used.
    _internal_types = {"int32": ScalarType(bitwidth=32, integral=True, boolean=False),
                       "int64": ScalarType(bitwidth=64, integral=True, boolean=False),
                       "float32": ScalarType(bitwidth=32, integral=False, boolean=False),
                       "float64": ScalarType(bitwidth=64, integral=False, boolean=False),
                       "pred32": ScalarType(bitwidth=32, integral=True, boolean=True),
                       "pred64": ScalarType(bitwidth=64, integral=True, boolean=True),
                       "fpred32": ScalarType(bitwidth=32, integral=False, boolean=True),
                       "fpred64": ScalarType(bitwidth=64, integral=False, boolean=True)}

    # These are only used for interface lookups, since we
    # don't distinguish numpy specific types internally.
    _standard_lookup = {np.int32: _internal_types["int32"],
                        np.int64: _internal_types["int64"],
                        np.float32: _internal_types["float32"],
                        np.float64: _internal_types["float64"],
                        np.float: _internal_types["float64"],
                        float: _internal_types["float64"]}

    def __init__(self, default_int64=True):

        self.lookup = self._standard_lookup.copy()

        if default_int64:
            self.lookup[int] = self._internal_types["int64"]
        else:
            self.lookup[int] = self._internal_types["int32"]

    def get_internal_type(self, item):
        is_array = isinstance(item, ArrayInput)
        is_internal_scalar_type = isinstance(item, ScalarType)
        if is_array:
            input_scalar_type = item.dtype
            if isinstance(input_scalar_type, ScalarType):
                dtype = item.dtype
            else:
                dtype = self.lookup.get(input_scalar_type)
        elif is_internal_scalar_type:
            dtype = input_scalar_type = item
        else:
            input_scalar_type = item
            dtype = self.lookup.get(input_scalar_type)
        if dtype is None:
            msg = f"Unable to map input parameter {input_scalar_type} to an internal type"
            raise KeyError(msg)
        if is_array:
            # wrap parameters
            dims = tuple(wrap_array_parameter(d) for d in item.dims)
            stride = wrap_array_parameter(item.stride)
            internal_type = ArrayInput(dims, dtype, stride)
        else:
            internal_type = dtype
        return internal_type

    @property
    def default_float(self):
        return self.lookup[float]

    @property
    def default_int(self):
        return self.lookup[int]


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
    def __init__(self, name, type_, is_added):
        self.name = name
        self.type_ = type_
        self.is_added = is_added

    def __eq__(self, other):
        assert isinstance(other, symbol)
        return (self.name == other.name
                and self.type_ == other.type_
                and self.is_added == other.is_added)

    def __ne__(self, other):
        assert isinstance(other, symbol)
        return (self.name != other.name
                or self.type_ != other.type_
                or self.is_added != other.is_added)

    def __hash__(self):
        return hash(self.name)

    @property
    def is_array(self):
        return isinstance(self.type_, ArrayInput)


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


class symboltable:
    def __init__(self, type_builder):
        self.symbols = {}
        self.added = set()
        self.type_builder = type_builder
        self.prefixes = {}  # prefix for adding enumerated variable names

    def wrap_name(self, name):
        if isinstance(name, ir.NameRef):
            return name
        elif isinstance(name, str):
            return ir.NameRef(name)
        else:
            msg = f"{name} is not a valid key for variable type lookup."
            raise TypeError(msg)

    def declares(self, name):
        name = self.wrap_name(name)
        return name in self.symbols

    def lookup(self, name):
        name = self.wrap_name(name)
        sym = self.symbols.get(name)
        if sym is None:
            msg = f"{name} is not a registered variable name."
            raise KeyError(msg)
        return sym

    def is_added_name(self, name):
        sym = self.lookup(name)
        return name.is_added

    @property
    def default_int(self):
        return self.type_builder.default_int

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
        return sym.is_array

    def matches(self, ref, value):
        """
        Check if ref is both registered and the existing definition matches value
        """
        if not self.declares(ref):
            return False
        existing = self.lookup(ref)
        return existing == value

    def make_symbol(self, name, type_, is_added):
        name = self.wrap_name(name)
        if self.declares(name):
            if not self.matches(name, type_):
                existing = self.lookup(name)
                msg = f"Existing symbol definition {existing} for {name} is incompatible with new definition" \
                      f"{type_}."
                raise KeyError(msg)
        else:
            sym = symbol(name, type_, is_added)
            self.symbols[name] = sym

    def make_unique_name(self, prefix):
        if isinstance(prefix, ir.NameRef):
            prefix = prefix.name
        if self.declares(prefix):
            gen = self._get_num_generator(prefix)
            name = f"{prefix}_{next(gen)}"
            while self.declares(name):
                name = f"{prefix}_{next(gen)}"
            name = ir.NameRef(name)
        else:
            # No need to rename
            name = prefix
        return name

    def add_var(self, name, type_, is_added):
        if not isinstance(type_, (ArrayInput, ArrayView, ScalarType)):
            type_ = self.type_builder.get_internal_type(type_)
        if is_added:
            name = self.make_unique_name(prefix=name)
        self.make_symbol(name, type_, is_added)

    def add_view(self, name, base, subscript, is_added):
        # Generally not intended
        if not isinstance(base, (ir.ArrayRef, ir.ViewRef)):
            # this could be a handle
            base = self.lookup(base)
        if not self.is_array(base):
            msg = f"{base.name} is not recognized as an array or view."
            raise TypeError(msg)
        # If the subscript expression is non-integral, this must be
        # caught elsewhere for now.
        view = ir.ViewRef(base, subscript)
        if is_added:
            name = self.make_unique_name(prefix=name)
        self.make_symbol(name, view, is_added)


def map_input_types_to_internal(by_type, canonical_types):
    """
    Map type parameterized name sets to use canonical types.
    This may merge type aliases that otherwise appear incompatible.

    """
    repl = {}
    for type_ in by_type:
        ct = canonical_types.get_internal_type(type_)
        repl[type_] = ct
    return repl


def assign_input_types(types, builder):
    """
    Map input types to unambiguous internal types.
    For example, int -> default_int_type

    """

    # map interface types to internal
    type_map = {}
    for type_ in types.values():
        ct = builder.get_internal_type(type_)
        type_map[type_] = ct

    # map each variable name to an internal type
    by_name = {}
    for name, type_ in types.items():
        if not isinstance(name, ir.NameRef):
            name = ir.NameRef(name)
        internal_type = type_map[type_]
        if name in by_name:
            first = by_name[name]
            if first != type_:
                msg = f"Duplicate type entry {first} and {type_} for name {name}"
                raise KeyError(msg)
        else:
            by_name[name] = internal_type
    return by_name


def map_to_internal_types(func, interface_types, type_builder):
    type_map = assign_input_types(interface_types, type_builder)
    func_name = func.get_name()
    # Check validity of type info
    for sym in func.get_symbols():
        name = ir.NameRef(sym.get_name())
        # checking only python reserved names thus far
        if name in reserved_names:
            if sym.is_assigned():
                raise NotImplementedError(f"Reassigning names used by the language itself is unsupported. "
                                          "{name} marked as assignment target")
        elif sym.is_parameter():
            if name not in type_map:
                raise TypeError(f"Missing type info for symbol {name} in function {func_name}")
    table = symboltable(type_builder)
    return table


def create_symbol_tables(src, filename, types_by_func, use_default_int64=True):
    type_builder = TypeBuilder(use_default_int64)
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
            # Only raise an error here for missing parameter type info.
            # Annotations are unsupported, because they are insufficient for array types.
            params = func.get_parameters()
            if func.get_parameters():
                raise ValueError(f"No type information provided for parameters: {params} of function {func_name}.")
        for sym in func.get_symbols():
            name = sym.get_name()
            if name in reserved_names:
                if sym.is_assigned():
                    raise NotImplementedError(f"Reassigning names used by the language itself is unsupported. "
                                              "{name} marked as assignment target")
        # Standardize type map for this function
        func_types = types_by_func.get(func_name)
        if func_types is None:
            raise ValueError(f"Missing type information for function {func_name} in file {filename}")
        func_table = map_to_internal_types(func, func_types, type_builder)
        for name, type_ in func_types.items():
            func_table.add_var(name, type_, is_added=False)
        tables[func_name] = func_table
    return tables
