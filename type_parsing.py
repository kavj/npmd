import numpy as np

import ir


class ArrayCreationInitializer:
    def __init__(self, dims, dtype, fill_value):
        self.dims = dims
        self.dtype = dtype
        self.fill_value = fill_value


class ArrayInputInitializer:
    def __init__(self, dims, dtype, stride):
        self.dims = dims
        self.dtype = dtype
        self.stride = stride

    @property
    def is_uniform(self):
        return self.stride == 0


class TypeBuilder:

    def __init__(self, default_int64=True):
        int32_type = ir.ScalarType(signed=True, boolean=False, integral=True, bitwidth=32)
        int64_type = ir.ScalarType(signed=True, boolean=False, integral=True, bitwidth=64)
        float32_type = ir.ScalarType(signed=True, boolean=False, integral=False, bitwidth=32)
        float64_type = ir.ScalarType(signed=True, boolean=False, integral=False, bitwidth=64)
        bool_type = ir.ScalarType(signed=True, boolean=True, integral=True, bitwidth=1)
        types = {np.int32: int32_type, np.int64: int64_type, float32_type: np.float32, float64_type: np.float64,
                 bool: bool_type}
        if default_int64:
            types[int] = int64_type
        else:
            types[int] = int32_type
        # Python floats are always double precision
        types[float] = float64_type
        self.types = types
        self.builders = {}

    @property
    def default_float(self):
        return self.types[float]

    @property
    def default_int(self):
        return self.types[int]

    def build_func(self, func: ir.Call):
        builder = self.builders.get(func.funcname)
        if builder is None:
            raise ValueError
        return builder(func.args, func.keywords)


def make_numpy_array(node: ir.Call, line):
    api_args = ("shape", "dtype", "order", "like")
    # only first 2 arguments supported
    args = node.args
    kws = node.keywords
    lookup = {}
    for name, arg in zip(api_args, args):
        lookup[name] = arg
    for name, arg in kws:
        if name in lookup:
            raise ValueError(f"Duplicate value for argument {name} in call to numpy.zeros, line: {line}")
        lookup[name] = arg
    if "dtype" not in lookup:
        lookup["dtype"] = np.float64
    if len(lookup) != 2:
        raise ValueError(f"Argument mismatch for call to numpy.zeros. Only shape and dtype parameters are supported.")
    if "shape" not in lookup:
        raise ValueError(f"Shape argument missing in call to numpy.zeros, line: {line}")
    func_name = node.funcname
    if func_name == "numpy.ones":
        fill_value = 1
    elif func_name == "numpy.zeros":
        fill_value = 0
    elif func_name == "numpy.empty":
        fill_value = None
    else:
        raise ValueError(f"No supported implementation for call to {node.funcname}")
    shape = lookup.get("shape")
    dtype = lookup.get("dtype")
    array = ArrayCreationInitializer(shape, dtype, fill_value)
    return array
