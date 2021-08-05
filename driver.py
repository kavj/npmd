import os
import sys

from errors import CompilerError
from ASTTransform import parse_file
from pretty_printing import pretty_printer

version = sys.version_info
# Python 2 can't parse a significant
# amount of this code, so error messages ignore it.
if sys.version_info.minor < 8:
    msg = f"Python 3.8 or above is required."
    raise RuntimeError(msg)


class CompilerContext:

    stages = []
    pretty_print = pretty_printer()

    def __init__(self, verbose=False, pretty_print_ir=False, pipeline=None):
        self.verbose = verbose
        self.pretty_print_ir_stages = pretty_print_ir
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            self.pipeline = [stage() for stage in CompilerContext.stages]

    def run_pipeline(self, source, file_name, type_map):

        with module_context():
            module, symbols = parse_file(source, file_name, type_map)

            funcs = module.functions

            if self.pretty_print_ir_stages:
                print(f"file name: {file_name}\n")
            for func in funcs:
                if self.pretty_print_ir_stages:
                    print(f"function: func.name\n")
                table = symbols[func.name]
                for stage in self.stages:
                    func = stage(func, table)
                    if self.pretty_print_ir_stages:
                        self.pretty_print(func, symbols)


def name_and_source_from_path(file_path):
    with open(file_path) as input:
        src = input.read()
    file_name = os.path.basename(file_path)
    return file_name, src


def compile(src, file_name, type_map, verbose=False, custom_pipeline=None):

    if verbose:
        if file_name:
            print(f"Compiling: {file_name}")
    try:
        tree, symbols = parse_file(src, file_name, type_map)
        functions = []
        for func in tree.functions:
            for stage in pipeline:
                func = stage(func, )

    except CompilerError:
        pass

