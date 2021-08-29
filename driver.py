import os
import sys
import typing

import numpy as np

import ir
import type_resolution as tr
from ASTTransform import parse_file
from canonicalize import NormalizePaths
from reaching_check import ReachingCheck
from dataclasses import dataclass
from errors import module_context
from utils import wrap_input
from pretty_printing import pretty_printer


version = sys.version_info
# Python 2 can't parse a significant
# amount of this code, so error messages ignore it.
if sys.version_info.minor < 8:
    raise RuntimeError(f"Python 3.8 or above is required.")


@dataclass(frozen=True)
class ArrayArg(ir.ValueRef):
    """
    Array argument. This ensures reaching definitions of array
    parameters are unambiguous.
    """
    spec: ir.ArrayInitSpec
    stride: typing.Optional[typing.Union[ir.NameRef, ir.IntConst]]


def get_scalar_type(input_type):
    if input_type == np.float32:
        return tr.Float32
    elif input_type == np.float64:
        return tr.Float64
    elif input_type == np.int32:
        return tr.Int32
    elif input_type == np.int64:
        return tr.Int64
    elif input_type == bool:
        return tr.BoolType
    else:
        msg = f"Supported types are {np.float32}, {np.float64}, {np.int32}, {np.int64}, {bool}, received {input_type}."
        raise ValueError(msg)


def make_array_arg_type(dims, dtype, stride=None):
    """
    Parameterized array type suitable for use as an argument.
    """
    dtype = get_scalar_type(dtype)
    dims = tuple(wrap_input(d) for d in dims)
    if stride is not None:
        stride = wrap_input(stride)
    spec = ir.ArrayInitSpec(dims, dtype, fill_value=None)
    return ArrayArg(spec, stride)


class CompilerContext:

    stages = [NormalizePaths, ReachingCheck]
    pretty_print = pretty_printer()

    def __init__(self, verbose=False, pretty_print_ir=False):
        self.verbose = verbose
        self.pretty_print_ir_stages = pretty_print_ir
        self.normalize_paths = NormalizePaths()
        self.reaching_check = ReachingCheck()
        self.pretty_print = pretty_printer()

    def run_pipeline(self, file_name, type_map):

        with module_context():
            module, symbols = parse_file(file_name, type_map)

            funcs = module.functions

            if self.pretty_print_ir_stages:
                print(f"file name: {file_name}\n")
            for index, func in enumerate(funcs):
                if self.pretty_print_ir_stages:
                    print(f"function: func.name\n")
                func = self.normalize_paths(func)
                self.reaching_check(func)
                funcs[index] = func
        return module, symbols


def name_and_source_from_path(file_path):
    with open(file_path) as src_stream:
        src = src_stream.read()
    file_name = os.path.basename(file_path)
    return file_name, src


def compile_module(file_name, type_map, verbose=False):

    if verbose:
        if file_name:
            print(f"Compiling: {file_name}")
    cc = CompilerContext(verbose=True, pretty_print_ir=True, pipeline=None)
    return cc.run_pipeline(file_name, type_map)

    # cc.run_pipeline(file_name, type_map)
    # for func in tree.functions:
    #    syms = symbols[func.name]
    #    for stage in cc.pipeline:
    #        func = stage(func, syms)
    #    functions.append(func)
