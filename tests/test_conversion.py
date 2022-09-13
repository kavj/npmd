import itertools
import pathlib
import pytest

# Todo: this could be made into a more complete class
from npmd.utils import statement_difference, statement_intersection
from npmd.ast_conversion import build_module_ir_and_symbols
from npmd.blocks import build_function_graph
from npmd.errors import CompilerError
from npmd.pretty_printing import DebugPrinter
from npmd.traversal import get_statement_lists
from tests.type_info import type_detail


# Todo: This needs a formal test suite rather than examples.
#       For now, make a second tuple to inject type info and extend
#       the printer to handle operator precedence and generation of annotations.

unconvertible = ('test_unpacking.py',)


def test_conversions():
    basepath = pathlib.Path(__file__).resolve().parent.parent.joinpath('tree_tests')
    for t in basepath.iterdir():
        if t.is_dir():
            # typically just __pycache__
            continue
        file_path = basepath.joinpath(t)
        func_types = type_detail[t.name]
        if file_path.name in unconvertible:
            with pytest.raises(CompilerError):
                build_module_ir_and_symbols(file_path, func_types)
            continue
        mod, symbols = build_module_ir_and_symbols(file_path, func_types)
        # Now test that conversions yield all nodes
        func = mod.functions[0]
        print(mod.name, func.name)
        tree_stmts = [stmt for stmt in itertools.chain(*get_statement_lists(func))]
        tree_stmts.append(func)
        graph = build_function_graph(func)
        graph_stmts = [stmt for stmt in graph.walk_nodes()]
        tree_stmt_ids = {id(stmt) for stmt in tree_stmts}
        graph_stmt_ids = {id(stmt) for stmt in graph_stmts}
        if tree_stmt_ids != graph_stmt_ids:
            only_in_tree = statement_difference(tree_stmts, graph_stmts)
            only_in_graph = statement_difference(graph_stmts, tree_stmts)
            printer = DebugPrinter()
            print('\n\nThe following appear only in the tree ir:')
            if only_in_tree:
                for stmt in only_in_tree:
                    printer.visit(stmt)
            else:
                print('    ...')
            print('\n\nThe following appear only in the graph ir:')

            if only_in_graph:
                for stmt in only_in_graph:
                    printer.visit(stmt)
            else:
                print('    ...')
            in_both = statement_intersection(tree_stmts, graph_stmts)

            print('\n\nThe following appear in both ir forms:')
            if in_both:
                for stmt in in_both:
                    printer.visit(stmt)
            else:
                print('    ...')

        assert tree_stmt_ids == graph_stmt_ids
