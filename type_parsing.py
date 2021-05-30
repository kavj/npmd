import typing
import numpy as np

from dataclasses import dataclass

import ir


@dataclass(frozen=True)
class ArrayType:
    ndims: int
    dtype: typing.Union[type, ir.ScalarType]
    uniform: bool
    fixed_shape: typing.Optional[typing.Tuple[int, ...]]

    def __post_init__(self):
        assert self.ndims > 0
        assert self.dtype in supported_scalar_types
        assert self.fixed_shape is None or len(self.fixed_shape) == self.ndims


@dataclass
class ArrayCreationRoutine:
    shape: typing.Tuple[typing.Union[ir.NameRef, ir.IntNode], ...]
    dtype: ir.ScalarType
    fill_value: typing.Optional[typing.Union[ir.NameRef, ir.Constant]]


# This will need some work for OS specific components..
# will expand later, bool is "special", it's usually broadcast to a larger type but assumed to be 1 bit
# as that's all that is needed to describe it and some lower level intrinsics and ir forms leverage this
supported_scalar_types = {np.int32: ir.ScalarType(signed=True, boolean=False, integral=True, bitwidth=32),
                          np.int64: ir.ScalarType(signed=True, boolean=False, integral=True, bitwidth=64),
                          np.float32: ir.ScalarType(signed=True, boolean=False, integral=False, bitwidth=32),
                          np.float64: ir.ScalarType(signed=True, boolean=False, integral=False, bitwidth=64),
                          bool: ir.ScalarType(signed=True, boolean=True, integral=True, bitwidth=1)}


def parse_ones(*args):
    argct = len(args)
    if not 0 < argct < 2:
        # order and like are not supported yet
        return
    shape = args[0]
    dtype = supported_scalar_types[np.float64] if len(args) != 2 else supported_scalar_types.get(args[1])
    fill_value = ir.IntNode(1) if dtype.integral else ir.FloatNode(1)
    return ArrayCreationRoutine(shape, dtype, fill_value)


def parse_zeros(*args):
    argct = len(args)
    if not 0 < argct < 2:
        # order and like are not supported yet
        return
    shape = args[0]
    dtype = supported_scalar_types[np.float64] if len(args) != 2 else supported_scalar_types.get(args[1])
    fill_value = ir.IntNode(0) if dtype.integral else ir.FloatNode(0)
    return ArrayCreationRoutine(shape, dtype, fill_value)


def parse_empty(*args):
    argct = len(args)
    if not 0 < argct < 2:
        # order and like are not supported yet
        return
    shape = args[0]
    dtype = supported_scalar_types[np.float64] if len(args) != 2 else supported_scalar_types.get(args[1])
    return ArrayCreationRoutine(shape, dtype, None)


array_creation_routines = {"zeros": parse_zeros,
                           "ones": parse_ones,
                           "empty": parse_empty}


def validate_typing(types: typing.Dict[str, typing.Union[type, ArrayType]]):
    early_validation_types = {}
    missing = set()
    for name, type_ in types.items():
        if isinstance(type_, type):
            t = supported_scalar_types.get(type_)
            if t is None:
                missing.add((name, type_))
            else:
                early_validation_types[name] = t
        elif isinstance(type_, ArrayType):
            early_validation_types[name] = ir.ArrayRef(type_.ndims, type_.dtype)
    return early_validation_types, missing


def parse_array_create(call_node: ir.Call, prefix="numpy"):
    call_name = call_node.funcname
    if call_name.startswith(prefix):
        call_name = call_name[len(prefix):]
        setup = array_creation_routines.get(call_name)
        return setup(call_node.args) if setup is not None else None
