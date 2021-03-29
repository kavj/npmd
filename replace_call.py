import ir

from utils import negate_condition


class ErrorObject:
    """
    Unless I find a better method for this, this allows us to delay logging an error until
    control returns to the statement handler, which can add positional information.
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class CallSpecialize:
    """
    Class to set up specialized call replacements. This is meant to be as simple as possible.
    Alternatively, it's possible to write a visitor that performs more involved replacements.

    This is forced to consider default arguments, because some builtins require it.
    It does not support * or ** signatures.

    """

    def __init__(self, name, args, repl, defaults, allows_keywords=False):
        self.name = name
        if len({*args}) < len(args):
            msg = f"Call lowering for {name} has duplicate argument fields, {args} received."
            raise ValueError(msg)
        self.args = args
        self.defaults = defaults
        self.allow_keywords = allows_keywords
        # Check no duplicates
        assert len(self.args) == len(args)
        # Check anything with a default also appears in args
        assert all(arg in self.args for (arg, value) in defaults.items())
        # repl must know how to take a dictionary of arguments and construct
        # a replacement node, suitable for downstream use
        self.replacement = repl

    @property
    def max_arg_count(self):
        return len(self.args)

    @property
    def default_count(self):
        return len(self.defaults)

    @property
    def min_arg_count(self):
        return len(self.args) - len(self.defaults)

    def validate_call(self, args, keywords):
        arg_count = len(args)
        kw_count = len(keywords)
        if not self.allow_keywords and kw_count > 0:
            return ErrorObject(f"Function {self.name} does not allow keyword arguments.")
        mapped = {}
        unrecognized = set()
        duplicates = set()
        missing = set()
        if arg_count + kw_count > self.max_arg_count:
            return ErrorObject(f"Function {self.name} has {self.max_arg_count} fields. "
                               f"{arg_count + kw_count} arguments were provided.")
        for field, arg in zip(self.args, args):
            mapped[field] = arg
        for kw, value in keywords:
            if kw not in self.args:
                unrecognized.add(kw)
            elif kw in mapped:
                duplicates.add(kw)
            mapped[kw] = value
        for field in self.args:
            if field not in mapped:
                if field in self.defaults:
                    mapped[field] = self.defaults[field]
                else:
                    missing.add(field)
        for u, v in unrecognized:
            return ErrorObject(f"unrecognized field {u} in call to {self.name}")
        return self.replacement(mapped)

    def validate_simple_call(self, args):
        # Check call with no keywords
        arg_count = len(args)
        mapped = {}
        if arg_count > self.max_arg_count:
            return ErrorObject(f"Signature for {self.name} accepts {self.max_arg_count} arguments. "
                               f"{arg_count} arguments received.")
        elif arg_count < self.min_arg_count:
            return ErrorObject(f"Signature for {self.name} expects at least {self.min_arg_count} arguments, "
                               f"{arg_count} arguments received.")
        for field, arg in zip(self.args, args):
            mapped[field] = arg
        for field, value in self.defaults.items():
            if field not in mapped:
                mapped[field] = value
        return mapped

    def __call__(self, node):
        # err (and warnings) could be logged, which might be better..
        # This should still return None on invalid mapping.
        if node.keywords:
            if not self.allow_keywords:
                return ErrorObject(f"{self.name} does not allow keywords.")
            else:
                mapped = self.validate_call(node.args, node.keywords)
                if isinstance(mapped, ErrorObject):
                    return mapped
                return self.replacement(mapped)
        else:
            mapped = self.validate_simple_call(node.args)
            if isinstance(mapped, ErrorObject):
                return mapped
            return self.replacement(mapped)


def EnumerateBuilder(mapping):
    return ir.Zip((ir.Counter(mapping["start"], None, ir.IntNode(1)), mapping["iterable"]))


def IterBuilder(mapping):
    return mapping["iterable"]


def ReversedBuilder(arg):
    if isinstance(arg, ir.Counter):
        if arg.stop is not None:
            node = ir.Counter(arg.stop, arg.start, negate_condition(arg.step))
        else:
            # must be enumerate
            return "Cannot reverse enumerate type"
    else:
        node = ir.Reversed(arg)
    return node


def ZipBuilder(node: ir.Call):
    """
    Zip takes an arbitrary number of positional arguments, something which we don't generally support.

    """
    assert (node.funcname == "zip")
    if node.keywords:
        return ErrorObject("Zip does not accept keyword arguments.")

    return ir.Zip(tuple(arg for arg in node.args))


def LenBuilder(mapping):
    """
    Stub to merge len() and .shape

    """
    return ir.ShapeRef(mapping["obj"], ir.IntNode(0))


def RangeBuilder(node: ir.Call):
    """
    Range is a special case and handled separately from other call signatures,
    since it has to accommodate two distinct orderings without the use of keywords.

    These are:
        range(stop)
        range(start, stop, step: optional)

    Unlike the CPython version, this is only supported as part of a for loop.

    """

    argct = len(node.args)
    if not (0 < argct <= 3):
        return None, "Range call with incorrect number of arguments"
    elif node.keywords:
        return None, "Range does not support keyword arguments"
    if argct == 1:
        repl = ir.Counter(ir.IntNode(0), node.args[0], ir.IntNode(1))
    elif argct == 2:
        repl = ir.Counter(node.args[0], node.args[1], ir.IntNode(1))
    else:
        repl = ir.Counter(node.args[0], node.args[1], node.args[2])
    return repl


builders = {"enumerate": CallSpecialize(name="enumerate",
                                        args=("iterable", "start"),
                                        repl=EnumerateBuilder,
                                        defaults={"start": ir.IntNode(0)},
                                        allows_keywords=True),

            "reversed": CallSpecialize(name="reversed",
                                       args=("object",),
                                       repl=ReversedBuilder,
                                       defaults={},
                                       allows_keywords=False),

            "iter": CallSpecialize(name="iter",
                                   args=("iterable",),
                                   repl=IterBuilder,
                                   defaults={},
                                   allows_keywords=False),

            "len": CallSpecialize(name="len",
                                  args=("obj",),
                                  repl=LenBuilder,
                                  defaults={},
                                  allows_keywords=False),

            "range": RangeBuilder,

            "zip": ZipBuilder,
            }


def replace_builtin_call(node: ir.Call):
    handler = builders.get(node.funcname)
    if handler is None:
        return node
    return handler(node)
