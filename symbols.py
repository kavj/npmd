from __future__ import annotations

import builtins
import itertools
import keyword
import numbers

import numpy as np

from functools import singledispatch, singledispatchmethod
from symtable import symtable

import ir

from visitor import walk_expr
from TypeInterface import ArrayInput, ScalarType

reserved_names = frozenset(set(dir(builtins)).union(set(keyword.kwlist)))


def wrap_constant(value):
    if isinstance(value, ir.Constant):
        return value
    if isinstance(value, bool):
        return ir.BoolNode(value)
    elif isinstance(value, numbers.Integral):
        # needed for numpy compatibility
        return ir.IntNode(value)
    elif isinstance(value, numbers.Real):
        # needed for numpy compatibility
        return ir.FloatNode(value)
    else:
        msg = f"{value} of type {type(value)} is not recognized as a constant."
        raise TypeError(msg)


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
def wrap_variable_name(name):
    msg = f"{name} cannot be used as an internal variable name. Expected a unicode string or internal name type."
    raise TypeError(msg)


@wrap_variable_name.register
def _(name: str):
    if name in reserved_names:
        msg = f"'{name}' is used by the Python language and may not be used as a local variable."
        raise ValueError(msg)
    return ir.NameRef(name)


@singledispatch
def wrap_array_parameter(param):
    if param is not None:
        msg = f"{param} cannot be coerced to a suitable array parameter type."
        raise TypeError(msg)


# return argument if already suitable wrapped
@wrap_array_parameter.register
def _(param: ir.NameRef):
    return param


@wrap_array_parameter.register
def _(param: ir.IntNode):
    return param


@wrap_array_parameter.register
def _(param: str):
    return wrap_variable_name(param)


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

    @property
    def is_integer(self):
        if isinstance(self.type_, ArrayInput):
            return False
        return self.type_.integral


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


def _make_reduce_dim_expr(expr, reduce_by):
    if expr.constant:
        value = expr.value - reduce_by
        reduced = wrap_constant(value)
    else:
        reduce_by = wrap_constant(reduce_by)
        reduced = ir.BinOp(expr, reduce_by, "-")
    return reduced


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
            name = ir.NameRef(prefix)
        return name

    # These return a name reference to the symbol name that is actually used.
    # In the case of is_added=False, this is the original name. Otherwise it
    # may be a unique variable name that instantiates some renaming or implementation binding.

    def add_var(self, name, type_, is_added):
        # run type check by default, to avoid internal array types
        # with Python or numpy types as parameters.
        type_ = self.type_builder.get_internal_type(type_)
        if is_added:
            name = self.make_unique_name(prefix=name)
        else:
            name = wrap_array_parameter(name)
            if self.declares(name):
                msg = f"Duplicate symbol declaration for variabl name '{name}'."
                raise ValueError(msg)
        self.make_symbol(name, type_, is_added)
        return name

    @singledispatchmethod
    def _get_view_type(self, ref, transpose, is_added):
        raise NotImplementedError

    @_get_view_type.register
    def _(self, ref: ir.NameRef, transpose, is_added):
        ref_type = self.lookup(ref)
        if ref_type.ndims == 1 and transpose:
            msg = f"Cannot transpose scalar view of a 1D array: {ref}"
            raise ValueError(msg)
        dims = ref_type.dims
        if len(dims) == 1:
            # view_type =
            pass
        else:
            pass

    @_get_view_type.register
    def _(self, ref: ir.Subscript, transpose, is_added):
        if not isinstance(ref.slice, ir.Slice):
            # single subscript. Since we don't allow binding slices to names
            # this must project onto a single leading value, assuming a valid input parameter.
            # Todo: This won't allow delayed evaluation of annotations, so have to unpack before this
            # Todo: ensure integral parameter
            raise ValueError
        ref_type = self.lookup(ref)

    def add_view(self, name, base, transpose, is_added):
        base_type = self.lookup(base)
        if not base_type.is_array:
            msg = f"{base_type.name} is not recognized as an array or view."
            raise TypeError(msg)
        for subexpr in walk_expr(base):
            if isinstance(subexpr, ir.NameRef):
                t = self.lookup(subexpr)
                if not t.is_integer:
                    msg = f"Non integral parameter {subexpr} cannot be used as an index or slice parameter."
                    raise ValueError(msg)
            elif isinstance(subexpr, ir.BinOp):
                if subexpr.op == "/":
                    msg = f"True divide generates non-integer expressions and therefore cannot be used as an " \
                          "index or slice parameter: {subexpr}."
                    raise ValueError(msg)
            elif isinstance(subexpr, ir.FloatNode):
                msg = f"Floating point types are not supported as part of a subscript expression. Encountered " \
                      f"{str(subexpr)}"
                raise ValueError(msg)
        dims = base.dims


def symbol_table_from_func(func, type_map, type_builder, filename):
    """
    Build an internally used symbol table from a Python function symtable and type information.

    func: Function symbol table symtable.symtable
    type_map: dict[name: interface_type]
    type_builder: builder to construct internal type info
    """
    func_name = func.get_name()
    if func.is_nested():
        raise ValueError(f"{func_name} in file {filename} appears as a nested scope, which is unsupported.")
    elif func.has_children():
        raise ValueError(f"{func_name} in file {filename} contains nested scopes, which are unsupported.")
    elif func.get_type() != "function":
        raise TypeError(f"{func_name} in file {filename} refers to a class rather than a function. This is "
                        f"unsupported.")
    missing = []
    for arg in func.get_parameters():
        # Check that all arguments have type info.
        if arg not in type_map:
            missing.append(arg)
    if missing:
        args = ", ".join(arg for arg in missing)
        msg = f"Function '{func.get_name()}' is missing type info for the following arguments: {args}."
        raise ValueError(msg)
    table = symboltable(type_builder)
    for name, type_ in type_map.items():
        table.add_var(name, type_, is_added=False)
    return table


def create_symbol_tables(src, filename, types_by_func, use_default_int64=True):
    type_builder = TypeBuilder(use_default_int64)
    tables = {}
    mod = symtable(src, filename, "exec")
    # extract names that correspond to functions
    for func in mod.get_children():
        name = func.get_name()
        types = types_by_func.get(name, ())
        tables[name] = symbol_table_from_func(func, types, type_builder, filename)
    return tables
