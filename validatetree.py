import builtins
import keyword
from dataclasses import dataclass
from functools import singledispatchmethod

import ir
from loopanalysis import LoopHeaderChecker
from visitor import VisitorBase, walk_assignments


@dataclass
class ErrorValue:
    msg: str
    pos: ir.Position


keywords = frozenset(set(keyword.kwlist))
builtin_names = frozenset(set(dir(builtins)))


def shadows_builtin_name(name):
    return name in keywords or name in builtin_names


def check_array_access():
    """
    Check for dimension count mismatches.
    Check that each subscript refers to an array input.

    """
    pass


def find_array_allocation():
    pass


def find_zero_dim_array():
    pass


def find_invalid_step():
    pass


def check_for_loop():
    pass


def find_aliased_array_names():
    pass


def find_bad_call_sigs():
    """

    """
    pass


def find_invalid_slices():
    pass


def is_supported_directive(name, args):
    """
    Check if matches simd or parallel directive

    """
    pass


def find_duplicate_args(func):
    """
    Haven't decided on keyword support...

    """
    args = set()
    dupes = set()
    for arg in func.args:
        if arg in args:
            dupes.add(arg)
        else:
            args.add(arg)
    if dupes:
        return dupes


def find_zero_step_size():
    pass


def find_unsafe_exprs(exprs):
    unsafe = set()
    for e in exprs:
        if isinstance(e, ir.BinOp):
            if e.op in ("/", "//", "/=", "//="):
                if e.right.is_constant:
                    pass
                    # if e.right.value == 0:
                    #    unsafe.add(e)
    return unsafe


class UntypedChecker(VisitorBase):

    def __call__(self, entry):
        self.errors = []
        self.visit(entry)

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Assign):
        is_tuple_target = isinstance(node, ir.Tuple)
        for target, iterable in walk_assignments(node):
            if is_tuple_target and not isinstance(target, ir.NameRef):
                msg = "Only unpacking to variable name targets is supported."
                pos = node.pos
                self.errors.append(ErrorValue(msg, pos))
            elif shadows_builtin_name(target):
                msg = "Builtin names and reserved keywords may not be assignment targets."
                pos = node.pos
                self.errors.append(ErrorValue(msg, pos))

    @visit.register
    def _(self, node: ir.CascadeAssign):
        value = node.value
        for target in node.targets:
            if not isinstance(target, ir.NameRef):
                msg = ""
                pos = node.pos
            elif shadows_builtin_name(target):
                msg = "Builtin names and reserved keywords may not be assignment targets."
                pos = node.pos
                self.errors.append(ErrorValue(msg, pos))

    @visit.register
    def _(self, node: ir.ForLoop):
        checker = LoopHeaderChecker(node)
        msg = ""
        if checker.tuple_valued_targets:
            msg = "For loop iterables must be fully unpacked"
        elif checker.unpacking_errors:
            msg = "Not enough values to unpack"
        elif checker.complex_assignments:
            msg = "Only name-valued targets are supported as unpacking targets."
        if msg != "":
            self.errors.append(ErrorValue(msg, node.pos))

    @visit.register
    def _(self, node: ir.Function):
        dupes = find_duplicate_args(node)
        if dupes:
            pos = ir.Position(0, 0, 0, 0)
            self.errors.append(ErrorValue("duplicate arguments in function", pos))
