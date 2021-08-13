import os.path

from ASTTransform import parse_file
from canonicalize import NormalizePaths
from reaching_check import ReachingCheck
from pretty_printing import pretty_printer


def run_tree_pipeline(pth, types):
    print("filepath: ", pth)
    reaching_check = ReachingCheck()
    filename = os.path.basename(pth)
    module, symbol_tables = parse_file(pth, types)
    repl = []
    pp = pretty_printer()
    normalize_paths = NormalizePaths()
    for func in module.functions:
        func = normalize_paths(func)
        pp(func, symbol_tables[func.name])
        print("\n\n")

    module.functions = repl
    unbound = []
    for func in module.functions:
        u = reaching_check(func)
        unbound.append(u)
    for u in unbound:
        u, raw, war = u
        for key in u:
            print("unbound:", key, "\n")
        for key in raw:
            print("used:", key, "\n")
        for key in war:
            print("overwritten after read:", key, "\n")
    return module, symbol_tables
