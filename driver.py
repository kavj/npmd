import os
import sys

from pathlib import Path

import ir

from ASTTransform import build_module_ir_and_symbols
from array_utils import find_array_view_parents
from ccodegen import codegen, Emitter
from canonicalize import NormalizePaths
from errors import CompilerError
from lowering import loop_lowering, array_offset_maker, remove_subarray_refs
from pretty_printing import pretty_formatter
from reaching_check import ReachingCheck
from utils import extract_name


version = sys.version_info
# Python 2 can't parse a significant
# amount of this code, so error messages ignore it.
if sys.version_info.minor < 8:
    raise RuntimeError(f"Python 3.8 or above is required.")


def make_setup_module(exts, path):
    emitter = Emitter(path.joinpath("setup.py"))

    with emitter.flush_on_exit():
        emitter.print_line("from pathlib import Path")
        emitter.print_line("from setuptools import setup, Extension")
        ext_strs = []
        for ext_name, file_name in exts.items():
            e = f"Extension('{ext_name}', [{file_name}]), "
            ext_strs.append(e)
        ext_strs = ", ".join(e for e in ext_strs)
        setup_str = f"setup(ext_modules=[{ext_strs}]):"
        emitter.print_line(setup_str)


def resolve_types(types):
    internal_types = {}
    for name, type_ in types.items():
        internal_type = ir.by_input_dtype.get(type_)
        if internal_type is None:
            msg = f"No internal type matches type {type_}."
            raise CompilerError(msg)
        internal_types[name] = type_
    return internal_types


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


# Todo: Add explicit entry points

def compile_module(file_path, types,  out_dir, verbose=False, print_result=True, check_unbound=True):
    out_dir = Path(out_dir).absolute()
    if "." in str(out_dir):
        msg = f"Expected a directory path, received: {out_dir} ."
        raise ValueError(msg)
    if not out_dir.exists():
        out_dir.mkdir(exist_ok=True, parents=True)
    if verbose:
        if file_path:
            print(f"Compiling: {file_path.name}:")
    modname = file_path.name
    modname, _ = os.path.splitext(modname)
    if not modname:
        msg = "No module specified"
        raise CompilerError(msg)
    mod_ir, symbols = build_module_ir_and_symbols(file_path, types)
    funcs = []
    norm_paths = NormalizePaths()
    rc = ReachingCheck()
    for func in mod_ir.functions:
        s = symbols.get(func.name)
        ll = loop_lowering(s)
        view_to_parents, offsets = find_array_view_parents(func, symbols[extract_name(func)])
        make_offsets = array_offset_maker(s, view_to_parents, offsets)
        remove_subarrays = remove_subarray_refs(s)
        func = norm_paths(func)
        if check_unbound:
            maybe_unbound = rc(func)
            if maybe_unbound:
                pf = pretty_formatter()
                mub = ", ".join(pf(m) for m in maybe_unbound)
                msg = f"The following variables are unbound along some paths. This is unsupported " \
                      f"to avoid tracking or ignoring UnboundLocal errors at runtime: {mub}."
                raise CompilerError(msg)
        func = ll(func)
        func = make_offsets(func)
        func = remove_subarrays(func)
        funcs.append(func)
        if print_result:
            from pretty_printing import pretty_printer
            pp = pretty_printer()
            pp(func, s)

    file_path = Path(out_dir).joinpath(f"{modname}Module.c")
    make_setup_module(exts={modname: file_path}, path=out_dir)
    codegen(out_dir, funcs, symbols, modname)

