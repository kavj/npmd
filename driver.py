import os
import sys
import typing

import numpy as np

from dataclasses import dataclass
from pathlib import Path

import ir

import type_resolution as tr

from ASTTransform import build_module_ir_and_symbols
from canonicalize import NormalizePaths
from errors import error_context, CompilerError
from pretty_printing import pretty_printer
from reaching_check import ReachingCheck
from type_inference import TypeInfer
from utils import wrap_input


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


def resolve_types(types):
    internal_types = {}
    for name, type_ in types.items():
        internal_type = tr.by_input_type.get(type_)
        if internal_type is None:
            msg = f"No internal type matches type {type_}."
            raise CompilerError(msg)
        internal_types[name] = type_
    return internal_types


class CompilerDriver:

    def __init__(self, types):
        self.build_module = ModuleBuilder()
        self.normalize_paths = NormalizePaths()
        self.reaching_check = ReachingCheck()
        self.pretty_print = pretty_printer(ctx_)
        self.ctx = ctx_

    def run_pipeline(self, file_name, type_map):
        with error_context():
            module = self.build_module(file_name)
            funcs = module.functions
            print(f"file name: {file_name}\n")
            for index, func in enumerate(funcs):
                func = self.normalize_paths(func)
                func_types = type_map[func.name]
                infer_types = TypeInfer(func_types)
                self.reaching_check(func)
                infer_types(func)
                # symbols[func.name].types = func_types
                funcs[index] = func
        return module

    def pretty_print_tree(self, module, func_name=None):
        with self.ctx.module_scope(module.name):
            if func_name is not None:
                with self.ctx.function_scope(func_name):
                    func = module.lookup(func_name)
                    self.pretty_print(func, self.ctx.current_function)
            else:
                for func in module.functions:
                    with self.ctx.function_scope(func.name):
                        self.pretty_print(func, self.ctx.current_function)


def name_and_source_from_path(file_path):
    with open(file_path) as src_stream:
        src = src_stream.read()
    file_name = os.path.basename(file_path)
    return file_name, src


# stub for now, since we may need to remake typed passes later
# per function or incorporate context management
def build_function_pipeline():
    pipeline = [NormalizePaths(),
                ReachingCheck()]
    return pipeline


def compile_module(file_path, types, verbose=False, print_result=False):
    pipeline = build_function_pipeline()
    if verbose:
        if file_path:
            print(f"Compiling: {file_name}:")
    mod_ir, symbols = build_module_ir_and_symbols(file_path, types)
    funcs = []
    for func in mod_ir.functions:
        for stage in pipeline:
            func = stage(func)
        funcs.append(func)
    if print_result:
        from pretty_printing import pretty_print
        pp = pretty_print()
        pp(mod_ir)
    return mod_ir

