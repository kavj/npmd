import builtins
import itertools
import keyword

import numpy as np

from symtable import symtable

import ir

from ArrayInterface import ArrayInput

reserved_names = frozenset(set(dir(builtins)).union(set(keyword.kwlist)))


class ArrayCreationInitializer:
    def __init__(self, dims, dtype, fill_value):
        self.dims = dims
        self.dtype = dtype
        self.fill_value = fill_value


class TypeLookup:
    def __init__(self):
        # partly intern basic types
        self.types = {"Int32": ir.int32(), "Int64": ir.int64(), "Float32": ir.float32(),
                      "Pred32": ir.pred32(), "Pred64": ir.pred64()}

    def lookup(self, type_):
        t = self.types.get(type_)
        if t is None:
            msg = f"No internal type matches name type name {type_}."
            raise KeyError(msg)
        return t


class TypeBuilder:

    def __init__(self, lookup, default_int64=True):

        types = {np.int32: lookup["int32"],
                 np.int64: lookup["int64"],
                 np.float32: lookup["float32"],
                 np.float64: lookup["float64"],
                 np.float: lookup["float64"],
                 float: lookup["float64"]}
        if default_int64:
            types[int] = lookup["int64"]
        else:
            types[int] = lookup["int32"]
        self.types = types

    def get_internal_type(self, item):
        is_array = isinstance(item, ArrayInput)
        is_internal_scalar_type = isinstance(item, ir.ScalarType)
        if is_array:
            input_scalar_type = item.dtype
            if isinstance(input_scalar_type, ir.ScalarType):
                dtype = item.dtype
            else:
                dtype = self.types.get(input_scalar_type)
        elif is_internal_scalar_type:
            dtype = input_scalar_type = item
        else:
            input_scalar_type = item
            dtype = self.types.get(input_scalar_type)
        if dtype is None:
            msg = f"Unable to map input parameter {input_scalar_type} to an internal type"
            raise KeyError(msg)
        if is_array:
            internal_type = ArrayInput(item.dims, dtype, item.stride)
        else:
            internal_type = dtype
        return internal_type

    @property
    def default_float(self):
        return self.types[float]

    @property
    def default_int(self):
        return self.types[int]


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
    def __init__(self, existing, type_builder):
        self.names = existing
        self.type_builder = type_builder
        self.prefixes = {}  # prefix for adding enumerated variable names

    def __contains__(self, item):
        if isinstance(item, ir.NameRef):
            item = item.name
        return item in self.names

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
        assert isinstance(name, ir.NameRef)
        sym = self.names.get(name)
        if sym is None:
            msg = f"No symbol table or type information available for variable name: {name}"
            raise KeyError(msg)
        return sym.is_array

    def add_view(self, name, base, subscript, added=False):
        if isinstance(name, str):
            name = ir.NameRef(name)
        if not self.is_array(name):
            msg = f"{name} is not recognized as an array or view."
            raise TypeError(msg)
        # If the subscript expression is non-integral, this must be
        # caught elsewhere for now.
        view = ir.ViewRef(base, subscript)

    def add_array(self, name, array_type):
        sym = symbol(name, array_type, is_added=True)
        if name in self.names:
            existing_sym = self.names[name]
            if existing_sym != sym:
                msg = f"Array name {name} shadows an existing name. For implementation reasons, arrays names are " \
                      f"restricted to a single definition per scope"
                raise KeyError(msg)
        else:
            self.names[name] = sym

    def add_view(self, name, array_type, base):
        sym = symbol(name, array_type)

    def make_unique_name(self, prefix, type_):
        gen = self._get_num_generator(prefix)
        name = f"{prefix}_{next(gen)}"
        while name in self.names:
            name = f"{prefix}_{next(gen)}"
        name = ir.NameRef(name)
        sym = symbol(name, type_, is_added=True)
        self.names[sym] = type_
        return name

    def lookup(self, name):
        sym = self.names.get(name)
        if name is None:
            msg = f"Missing symbol table entry for {name}"
            raise KeyError(msg)
        return sym


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
    for type_ in types:
        ct = builder.get_internal_type(type_)
        type_map[type_] = ct

    # map each variable name to an internal type
    by_name = {}
    for type_, names in types.items():
        internal_type = type_map[type_]
        for name in names:
            if name in by_name:
                first = by_name[name]
                msg = f"Duplicate type entry {first} and {type_} for name {name}"
                raise KeyError(msg)
            if isinstance(internal_type, ArrayInput):
                # Check for implicit parameters
                dims = internal_type.dims
                stride = internal_type.stride
                for dim in dims:
                    if isinstance(dim, str):
                        # check consistency
                        if dim in by_name:
                            prior_type = by_name[dim]
                            if not prior_type.integral:
                                msg = f"Type conflict. Existing type info for array dim parameter {dim} is not integral."
                                raise ValueError(msg)
                        else:
                            # prior takes precedence over default int here
                            by_name[dim] = builder.default_int
                if isinstance(stride, str):
                    if stride in by_name:
                        prior_type = by_name[stride]
                        if not prior_type.integral:
                            msg = f"Existing type info stride parameter {stride} of array {name} is not integral."
                            raise ValueError(msg)
                    else:
                        by_name[stride] = builder.default_int

            by_name[name] = internal_type
    wrapped = {}
    for name, sym in by_name.items():
        wrapped[ir.NameRef(name)] = sym
    return wrapped


def map_types_by_func(func, interface_types, type_builder):
    type_map = assign_input_types(interface_types, type_builder)
    func_name = func.get_name()
    # Check validity of type info
    for sym in func.get_symbols():
        name = sym.get_name()
        # checking only python reserved names thus far
        if name in reserved_names:
            if sym.is_assigned():
                raise NotImplementedError(f"Reassigning names used by the language itself is unsupported. "
                                          "{name} marked as assignment target")
        elif name not in type_map:
            raise TypeError(f"Missing type info for symbol {name} in function {func_name}")
    table = symboltable(type_map, type_builder)
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
        func_table = map_types_by_func(func, func_types, type_builder)
        tables[func_name] = func_table
    return tables
