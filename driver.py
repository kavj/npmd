import os
import sys

from pathlib import Path

from pybind_gen import gen_module, gen_setup, Emitter

from ast_conversion import build_module_ir_and_symbols
from blocks import build_function_graph, graph_to_pydot, render_pydot, patch_missing_return
from dead_code import remove_unreachable_blocks, remove_statements_following_terminals, remove_trivial_continues, strip_continues
from errors import CompilerError
from liveness import check_all_assigned, dump_live_info, find_live_in_out, remove_dead_statements
from loop_simplify import lower_loops, sanitize_loop_iterables
from pprint import pformat
from pretty_printing import PrettyPrinter
from type_checks import dump_symbol_type_info, TypeInference


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

def compile_module(file_path, types,  out_dir, verbose=False, print_result=True, allow_inference=True, debug=False):
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
        TypeInference(func_symbols).visit(func)
        func = remove_dead_statements(func, func_symbols)
        remove_statements_following_terminals(func)
        # more aggressive
        strip_continues(func)
        sanitize_loop_iterables(func, func_symbols)
        func_graph = build_function_graph(func)
        lower_loops(func.body, func_symbols, func_graph)
        func = remove_dead_statements(func, func_symbols)
        func_graph = build_function_graph(func)
        func_graph = remove_unreachable_blocks(func_graph)

        patch_missing_return(func_graph)

        if debug:
            file_path_ = Path(out_dir)
            assert file_path_.is_dir()
            file_dir = file_path_.parent
            dot_name = graph_to_pydot(func_graph, func.name, modname, file_dir)
            render_path = Path(out_dir)
            render_pydot(dot_name, render_path)
            func_graph = remove_unreachable_blocks(func_graph)
            patch_missing_return(func_graph)

            liveness = find_live_in_out(func_graph)
            check_all_assigned(func_graph)
            dump_live_info(liveness, ignores=func.args)
            dot_name = graph_to_pydot(func_graph, func.name + '_after_dce', modname, file_dir)
            render_path = Path(out_dir)
            render_pydot(dot_name, render_path)

        if print_result:
            print('type info..')
            dump_symbol_type_info(func_symbols)
            print()
            pp = PrettyPrinter()
            pp(func, func_symbols)

        funcs.append(func)

    gen_module(out_dir, modname, funcs, symbols)
    gen_setup(out_dir, modname)
