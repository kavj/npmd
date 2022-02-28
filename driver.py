import os
import sys

from pathlib import Path

from pbgen import gen_module, gen_setup, Emitter

from ASTTransform import build_module_ir_and_symbols
from canonicalize import NormalizePaths, patch_return
from errors import CompilerError
from lowering import LoopLowering
from pretty_printing import pretty_formatter
from pprint import pformat
from reaching_check import ReachingCheck

version = sys.version_info
# Python 2 can't parse a significant
# amount of this code, so error messages ignore it.
if sys.version_info.minor < 8:
    raise RuntimeError(f"Python 3.8 or above is required.")


def make_setup_module(exts, path):
    emitter = Emitter(path.joinpath("setup.py"))

    with emitter.flush_on_exit():
        # Todo: standardize this with nodes rather than ad hoc printing
        emitter.print_line("from setuptools import setup")
        emitter.print_line("from make_extensions import make_extensions")
        emitter.print_line("from pathlib import Path")
        emitter.print_line("import numpy")
        emitter.print_line(line=f"exts={pformat(exts)}")
        emitter.print_line(line=f"processed_exts=make_extensions(exts)")
        # temporary measure
        utils_include = str((Path.cwd().joinpath("runtime")))

        setup_str = f'setup(name="{path.name}", ext_modules=processed_exts, include_dirs=[numpy.get_include(), \"{utils_include}\"])'
        emitter.print_line(setup_str)


def name_and_source_from_path(file_path):
    with open(file_path) as src_stream:
        src = src_stream.read()
    file_name = os.path.basename(file_path)
    return file_name, src


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
        func_symbols = symbols.get(func.name)
        loop_lowering = LoopLowering(func_symbols)
        func = norm_paths(func)
        if check_unbound:
            maybe_unbound = rc(func)
            if maybe_unbound:
                pf = pretty_formatter()
                mub = ", ".join(pf(m) for m in maybe_unbound)
                msg = f"The following variables are unbound along some paths. This is unsupported " \
                      f"to avoid tracking or ignoring UnboundLocal errors at runtime: {mub}."
                raise CompilerError(msg)
        func = loop_lowering.visit(func)
        func = patch_return(func, func_symbols)

        funcs.append(func)
        if print_result:
            from pretty_printing import pretty_printer
            pp = pretty_printer()
            pp(func, func_symbols)

    gen_module(out_dir, modname, funcs, symbols)
    gen_setup(out_dir)
