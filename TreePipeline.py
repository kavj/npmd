from ASTTransform import build_module_ir
from Canonicalize import RemoveUnreachable, RemoveTrailingContinues, MergePaths
from reachingcheck import ReachingCheck


def run_tree_pipeline(pth):
    print("filepath: ", pth)
    with open(pth) as r:
        r = r.read()
    mod = build_module_ir(r)
    r = RemoveUnreachable()
    rtc = RemoveTrailingContinues()
    rc = ReachingCheck()
    mp = MergePaths()
    mod = r(mod)
    mod = mp(mod)
    mod = rtc(mod)
    unbound = []
    for func in mod.funcs:
        u = rc(func)
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
