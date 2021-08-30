import builtins
import keyword

import ir


keywords = frozenset(set(keyword.kwlist))
builtin_names = frozenset(set(dir(builtins)))


# Todo: This is marked for rewrite. The previous version was completely out of date.


def shadows_builtin_name(name):
    return name in keywords or name in builtin_names


def check_argument_errors(node: ir.Function):
    errors = []
    seen = set()
    for arg in node.args:
        if shadows_builtin_name(arg):
            msg = f"Argument {arg.name} shadows a builtin name."
            errors.append(msg)
        if arg in seen:
            msg = f"Duplicate argument name {arg.name}."
            errors.append(msg)
        seen.add(arg)
    if not errors:
        errors = None
    return errors
