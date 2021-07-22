import os.path

from ASTTransform import build_module_ir
from Canonicalize import RemoveContinues, MergePaths
from reachingcheck import ReachingCheck
from printing import printtree


def run_tree_pipeline(pth, types):
    print("filepath: ", pth)
    with open(pth) as r:
        r = r.read()
    remove_continues = RemoveContinues()
    reaching_check = ReachingCheck()
    merge_paths = MergePaths()
    filename = os.path.basename(pth)
    mod = build_module_ir(r, filename, types)
    repl = []
    P = printtree()
    for func in mod.funcs:
        P(func)
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

    mod.funcs = repl
    unbound = []
    for func in mod.funcs:
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
    return mod
