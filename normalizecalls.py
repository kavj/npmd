import ir

from utils import negate_condition


def repl_enumerate(args):
    """
    This turns a Call node, calling enumerate with arbitrary argument setup into an IR enumerate representation.
    Since we have to distinguish the counter component on its own anyway during unpacking analysis, it's easier
    to treat this as a 2 element zip. 

    """
    posargct = len(args)
    if not (1 <= posargct <= 2):
        msg = f"enumerate must have 1 or 2 arguments, received {posargct}"
        raise ValueError(msg)
    entries = {}
    fields = ("iterable", "start")
    for key, arg in zip(fields, args):
        entries[key] = arg
    iterable = entries["iterable"]
    if "start" in entries:
        start = entries["start"]
    else:
        start = ir.IntNode(0)
    c = ir.Counter(start, None, ir.IntNode(1))
    return ir.Zip((c, iterable))


def repl_range(args):
    """
    This transforms an arbitrary range call into a range node with three argument form
    
    """
    # Check number of unnamed args
    argct = len(args)
    if not (0 < argct <= 3):
        raise ValueError("Range call with incorrect number of arguments")
    if argct == 1:
        repl = ir.Counter(ir.IntNode(0), args[0], ir.IntNode(1))
    elif argct == 2:
        repl = ir.Counter(args[0], args[1], ir.IntNode(1))
    else:
        repl = ir.Counter(args[0], args[1], args[2])
    return repl


def repl_reversed(arg):
    if isinstance(arg, ir.Counter):
        # definitely can't reverse enumerate
        if arg.stop is None:
            raise TypeError("Enumerate is not reversible.")
        node = ir.Counter(arg.stop, arg.start, negate_condition(arg.step))
    else:
        node = ir.Reversed(arg)
    return node


def repl_zip(args):
    return ir.Zip(tuple(arg for arg in args))


def repl_len(arg):
    """
    Stub to merge len() and .shape

    """
    return ir.Subscript(ir.ShapeRef(arg), ir.IntNode(0))


def replace_builtin_call(funcname, args, keywords):
    if funcname == "zip":
        expr = repl_zip(args)
    elif funcname == "enumerate":
        expr = repl_enumerate(args)
    elif funcname == "reversed":
        if len(args) != 1:
            raise ValueError("Reversed can only take one argument.")
        expr = repl_reversed(args)
    elif funcname == "range":
        expr = repl_range(args)
    elif funcname == "len":
        if len(args) != 1:
            raise ValueError("Incorrect number of arguments")
        return ir.Subscript(ir.ShapeRef(args[0]), ir.IntNode(0))
    else:
        expr = ir.Call(funcname, args, ())
    return expr
