import os.path

from ASTTransform import build_module_ir
from Canonicalize import RemoveContinues, MergePaths
from reachingcheck import ReachingCheck
from pretty_printing import pretty_printer


def run_tree_pipeline(pth, types):
    print("filepath: ", pth)
    with open(pth) as r:
        r = r.read()
    remove_continues = RemoveContinues()
    reaching_check = ReachingCheck()
    merge_paths = MergePaths()
    filename = os.path.basename(pth)
    module, symbol_tables = build_module_ir(r, filename, types)
    repl = []
    pp = pretty_printer()
    for func in module.functions:
        pp(func, symbol_tables[func.name])
        print("\n\n")
        # func = merge_paths(func)
        # P(func)
        # print("\n\n")
        # func = fold_constant_expressions(func)
        # P(func)
        # print("\n\n")
        # func = merge_paths(func)
        # P(func)
        # print("\n\n")
        # func = fold_constant_expressions(func)
        # P(func)
        # print("\n\n")
        # func = remove_continues(func)
        # repl.append(func)

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
