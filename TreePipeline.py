import ir

from ASTTransform import build_module_ir
from Canonicalize import RemoveContinues, MergePaths
from folding import fold_constant_expressions
from reachingcheck import ReachingCheck
# from lowering import loop_lower
from varying import varying_check


def run_tree_pipeline(pth):
    print("filepath: ", pth)
    with open(pth) as r:
        r = r.read()
    remove_continues = RemoveContinues()
    reaching_check = ReachingCheck()
    merge_paths = MergePaths()
    # ll = loop_lower("lindex")
    mod = build_module_ir(r)
    vv = varying_check()
    repl = []
    varies = []
    for func in mod.funcs:
        func = merge_paths(func)
        func = fold_constant_expressions(func)
        func = merge_paths(func)
        func = fold_constant_expressions(func)
        func = remove_continues(func)
        # func = ll(func)
        varies = (vv(func, {ir.NameRef("a"), ir.NameRef("b")}, {ir.NameRef("a"), ir.NameRef("b")}))
        repl.append(func)

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
    for func in mod.funcs:
        print("varying")
        for stuff in varies:
            print(stuff)

    return mod
