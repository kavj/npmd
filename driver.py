import os
import sys

from pathlib import Path

from pybind_gen import gen_module, gen_setup, Emitter

from analysis import check_all_declared, DeclTracker
from ast_conversion import build_module_ir_and_symbols
from canonicalize import patch_return
from errors import CompilerError
from loop_simplify import LoopLowering, remove_unreachable
from lvn import remove_dead, optimize_statements
from pprint import pformat
from type_checks import TypeInference


version = sys.version_info
if sys.version_info.major != 3 or sys.version_info.minor < 8:
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

def compile_module(file_path, types,  out_dir, verbose=False, print_result=True, allow_inference=True):
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
    for func in mod_ir.functions:
        func_symbols = symbols.get(func.name)
        loop_lowering = LoopLowering(func_symbols)
        tracker = DeclTracker()
        with tracker.scope():
            for arg in func.args:
                tracker.declare(arg)
            check_all_declared(func.body, tracker)
        func.body = remove_unreachable(func.body)
        func = remove_dead(func, func_symbols)
        func = optimize_statements(func, func_symbols)
        TypeInference(func_symbols).visit(func)
        func = loop_lowering.visit(func)
        func = remove_dead(func, func_symbols)
        func = patch_return(func, func_symbols)

        if print_result:
            from pretty_printing import PrettyPrinter
            pp = PrettyPrinter()
            pp(func, func_symbols)

        funcs.append(func)

    gen_module(out_dir, modname, funcs, symbols)
    gen_setup(out_dir, modname)
