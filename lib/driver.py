import os
import sys

from pathlib import Path

from lib.ast_conversion import build_module_ir_and_symbols
from lib.blocks import build_function_graph, render_dot_graph, render_dominator_tree
from lib.branch_simplify import refactor_branches
from lib.canonicalize import add_trivial_return, expand_in_place_assignments, normalize_array_initializers, lower_loops
from lib.errors import CompilerError
from lib.liveness import check_all_assigned, drop_unused_symbols, dump_live_info, find_live_in_out, remove_unreachable_blocks, remove_unreachable_statements
from lib.pretty_printing import PrettyPrinter
from lib.pybind_gen import gen_module, gen_setup
from lib.symbol_table import dump_symbol_type_info
from lib.type_checks import infer_types, TypeHelper


version = sys.version_info
if sys.version_info.major != 3 or sys.version_info.minor < 8:
    raise RuntimeError(f"Python 3.8 or above is required.")


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
        add_trivial_return(func)
        refactor_branches(func, symbols)
        remove_unreachable_statements(func, func_symbols)
        normalize_array_initializers(func, func_symbols)
        infer_types(func, func_symbols)
        typer = TypeHelper(func_symbols)
        expand_in_place_assignments(func, typer)
        lower_loops(func, func_symbols)
        remove_unreachable_statements(func, func_symbols)
        refactor_branches(func, symbols)
        remove_unreachable_statements(func, func_symbols)
        drop_unused_symbols(func, func_symbols)
        render_path = Path(out_dir)
        assert render_path.is_dir()
        func_graph = build_function_graph(func)
        liveness = find_live_in_out(func_graph)
        if debug:
            func_graph = remove_unreachable_blocks(func_graph)
            check_all_assigned(func_graph)
            dump_live_info(liveness, ignores=func.args)
            render_dot_graph(func_graph.graph, func_graph.func_name, render_path)
            render_dominator_tree(func_graph, render_path)

        if print_result:
            print('type info..')
            dump_symbol_type_info(func_symbols)
            print()
            pp = PrettyPrinter()
            pp(func, func_symbols)

        funcs.append(func)

    gen_module(out_dir, modname, funcs, symbols)
    gen_setup(out_dir, modname)
