import os
import sys

from pathlib import Path

from lib.cfg_builder import build_module_ir
from lib.blocks import render_dot_graph
from lib.loop_lowering import preheader_rename_parameters
from lib.errors import CompilerError
from lib.branch_normalize import hoist_terminals
from lib.liveness import check_for_maybe_unbound_names, remove_unreachable_blocks
from lib.pretty_printing import PrettyPrinter
from lib.symbol_table import dump_symbol_type_info


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
    mod_ir = build_module_ir(file_path, types)
    funcs = []
    for func in mod_ir.functions:

        pp = PrettyPrinter()

        remove_unreachable_blocks(func)
        preheader_rename_parameters(func)
        hoist_terminals(func)
        render_dot_graph(func.graph, 'test_graph', Path.cwd())
        render_path = Path(out_dir)
        unbound_names, _ = check_for_maybe_unbound_names(func)
        if unbound_names:
            print(f'unbound pairs for "{func.name}":')
            for pair in unbound_names.items():
                print(pair)
        # if debug:
        #    func_graph = remove_unreachable_blocks(func_graph)
        #    check_all_assigned(func_graph)
        #    dump_live_info(liveness, ignores=func.args)
        render_dot_graph(func.graph, func.name, render_path)
        #    render_dominator_tree(func_graph, render_path)

        if print_result:
            print('type info..')
            dump_symbol_type_info(func.symbols)
            print()
            pp = PrettyPrinter()
            pp(func)

        funcs.append(func)

    # gen_module(out_dir, modname, funcs, symbols)
    # gen_setup(out_dir, modname)
