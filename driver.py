import numbers
import os
import sys
import typing

import numpy as np

from dataclasses import dataclass
from pathlib import Path

import ir

import type_interface as ti
import type_resolution as tr

from ASTTransform import build_module_ir_and_symbols
from ccodegen import codegen
from canonicalize import NormalizePaths
from errors import error_context, CompilerError
from lowering import loop_lowering
from pretty_printing import pretty_printer
from reaching_check import ReachingCheck
from utils import wrap_input


version = sys.version_info
# Python 2 can't parse a significant
# amount of this code, so error messages ignore it.
if sys.version_info.minor < 8:
    raise RuntimeError(f"Python 3.8 or above is required.")


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


def compile_module(file_path, types, verbose=False, print_result=True, out=None):
    # pipeline = build_function_pipeline()
    if verbose:
        if file_path:
            print(f"Compiling: {file_name}:")
    modname = file_path.name
    if not modname:
        msg = "No module specified"
        raise CompilerError(msg)
    mod_ir, symbols = build_module_ir_and_symbols(file_path, types)
    funcs = []
    norm_paths = NormalizePaths()
    # rc = ReachingCheck()
    for func in mod_ir.functions:
        s = symbols.get(func.name)
        ll = loop_lowering(s)
        func = norm_paths(func)
        func = ll(func)
        funcs.append(func)
        if print_result:
            from pretty_printing import pretty_printer
            pp = pretty_printer()
            pp(func, s)
    if out is None:
        # try in same folder
        out = Path.cwd()
    codegen(out, funcs, symbols, modname)
