from contextlib import contextmanager
from functools import singledispatchmethod

import ir
import utils

from errors import CompilerError
from pretty_printing import pretty_formatter, format_error
from utils import unpack_iterated, extract_name
from type_resolution import ExprTypeInfer
from visitor import StmtVisitor


def find_subscript_len_expr(expr: ir.Subscript):
    # nested slicing is unsupported for simplicity
    if not isinstance(expr.value, ir.NameRef):
        pf = pretty_formatter()
        msg = f"Nested subscripts are unsupported: {pf(expr)}."
        raise CompilerError(msg)

    if isinstance(expr.index, ir.Slice):
        # so the last operation(s) are offsets or striding
        stop_expr = expr.index.stop if expr.index.stop == ir.Length(expr.value) \
            else ir.Min(ir.Length(expr.value), expr.index.stop)
        start_expr = expr.index.start if expr.index.start == ir.Zero \
            else ir.Max(expr.index.start, ir.Zero)
        # possible to run an optimizer here
        len_expr = stop_expr if start_expr == ir.Zero else ir.SUB(stop_expr, start_expr)
        if expr.index != ir.One:
            base_count = ir.FLOORDIV(len_expr, expr.index.step)
            add_fringe = ir.MOD(len_expr, expr.index.step)
            len_expr = ir.Select(add_fringe, base_count, ir.ADD(base_count, ir.One))
    elif isinstance(expr.index, ir.Tuple):
        msg = f"Tuple based multi-subscripting is not yet supported."
        raise NotImplementedError(msg)
    else:
        len_expr = ir.SingleDimRef(expr.value, ir.One)
    return len_expr


class ArrayViewCollect(StmtVisitor):

    @contextmanager
    def assign_context(self):
        self.targets = {}
        yield
        self.targets = None

    def __init__(self, syms):
        self.type_infer = ExprTypeInfer(syms)
        self.targets = None

    def __call__(self, node):
        with self.assign_context():
            self.visit(node)
            return self.targets

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.ForLoop):
        pf = pretty_formatter()
        for target, iterable in unpack_iterated(node.target, node.iterable):
            if not isinstance(target, ir.NameRef):
                msg = f"Only simple named variables and fully unpackable tuples of simple named variables" \
                      f" are supported as for loop targets for loops '{pf(target)}'."
                raise CompilerError(msg)
            target_type = self.type_infer(target)
            # determine set of value_types that are compatible with this target
            value_type = self.type_infer(iterable)
            if isinstance(target_type, ir.ArrayType):
                if value_type != ir.ArrayType(target_type.ndims + 1, target_type.dtype):
                    msg = f"Target variable '{pf(target)}' with type '{pf(target_type)}' is incompatible with " \
                          f"iterated value '{pf(iterable)}' of type '{pf(value_type)}'."
                    raise CompilerError(msg)
                existing = self.targets.get(target, iterable)
                if existing is not None and existing != iterable:
                    pf = pretty_formatter()
                    msg = f"Array to sub-array relationships must be unique, found target: " \
                          f"{pf(target)} from {pf(existing)} and {pf(iterable)} ."
                    raise CompilerError(msg)

            elif isinstance(target_type, ir.ScalarType):
                if isinstance(iterable, ir.AffineSeq):
                    if not isinstance(target_type, (ir.IntegerType, ir.UnsignedType)):
                        msg = f"Expected {pf(target)} to have integer type, matching" \
                              f"affine sequence {pf(iterable)}."
                        raise CompilerError(msg)
                elif value_type != ir.ArrayType(ndims=1, dtype=target_type):
                    formatted = format_error(msg="Iterator type is incompatible with target type.",
                                             named={value_type: target_type},
                                             exprs=None, pos=node.pos)
                    raise CompilerError(formatted)

            if isinstance(target_type, ir.ArrayType):
                if not isinstance(target, ir.NameRef):
                    # unsupported, can't assign to subscripted subarray that is a sub-array
                    if isinstance(target, ir.Subscript):
                        msg = f"Assignments to sliced array views are unsupported: line{node.pos.line_begin}."
                    else:
                        msg = f"Augmented assignments to array views are unsupported: line{node.pos.line_begin}"
                    raise CompilerError(msg)
                existing_base = self.targets.get(target, iterable)
                if existing_base is not None and existing_base != iterable:
                    pf = pretty_formatter()
                    msg = f"Array to sub-array relationships must be unique, found target: " \
                          f"{pf(target)} from {pf(existing_base)} and {pf(iterable)} ."
                    raise CompilerError(msg)
                self.targets[target] = iterable

    @visit.register
    def _(self, node: ir.Assign):
        target_type = self.type_infer(node.target)
        value_type = self.type_infer(node.value)
        existing = self.targets.get(node.target)
        if isinstance(node.value, ir.ArrayInitializer):
            if existing is not None:
                pf = pretty_formatter()
                msg = f"Binding of array allocations to named variables must be unique: received " \
                      f"'{pf(node.value)}' and '{pf(existing)}', line: {node.pos.line_begin}."
                raise CompilerError(msg)
            self.targets[node.target] = node.value
        elif isinstance(target_type, ir.ArrayType):
            # only supported in loops for now
            if not isinstance(node.target, ir.NameRef):
                # unsupported, can't assign to subscripted subarray that is a sub-array
                if isinstance(node.target, ir.Subscript):
                    msg = f"Assignments to sliced array views are unsupported: line{node.pos.line_begin}."
                else:
                    msg = f"Augmented assignments to array views are unsupported: line{node.pos.line_begin}"
                raise CompilerError(msg)
            value = node.value
            if isinstance(value, ir.Subscript):
                # get named ref in case this is used with different subscripts.
                value = value.value
            self.targets[node.target] = value


def find_array_view_parents(node, syms):
    avt = ArrayViewCollect(syms)
    view_to_parent = avt(node)
    infer_type = ExprTypeInfer(syms)
    # now make a corresponding offset variable
    # for each mapping of parent view to subview
    offset_names = {}
    for target in view_to_parent:
        name = extract_name(target)
        type_ = infer_type(target)
        offset_names[target] = syms.make_unique_name_like(f"{name}_i", type_)
    return view_to_parent, offset_names


def find_array_bases(view_to_expr):
    """

    :param view_to_expr: map of view name to constructing expression
    :return:
    """

    # now find initial bases,
    # these are known to derive from the same parent views
    # since the iterable: subview relationship is constrained to unique pairings
    # Further issues are basically solved by checking uniformity of iteration bounds.
    view_to_base = {}

    # first check for cycles
    seen = set()

    for view, parent in view_to_expr.items():
        # check for cycles
        seen.clear()
        view_name = ir.NameRef(extract_name(view))
        seen.add(view_name)
        while view in view_to_expr:
            view = view_to_expr[view]
            view_name = ir.NameRef(extract_name(view))
            if view_name in seen:
                pf = pretty_formatter()
                msg = f"Detected array view cycle or self reference for: {pf(view_name)}."
                raise CompilerError(msg)

    # find array allocations
    for derived, parent in view_to_expr.items():
        if not isinstance(parent, ir.ArrayInitializer):
            view_to_base[derived] = parent

    # map each view to corresponding base
    for view, parent_view in view_to_expr.items():
        enqueued = set()
        while parent_view in view_to_expr:
            enqueued.add(parent_view)
            parent_view = view_to_expr[parent_view]
        for e in enqueued:
            view_to_base[e] = parent_view

    return view_to_base


def make_offset_names(view_to_parent, syms):
    # Now we need to make an offset variable for each
    # array ref that is not an outermost parent view.
    offset_names = {}

    for view, parent in view_to_parent.items():
        raw_name = extract_name(view)
        augmented = f"{raw_name}_offset"
        offset_names[view] = syms.make_unique_name_like(augmented)

    return offset_names


def make_offset_coordinate_expr(expr, view_to_parent, offset_names):
    # first get a slice hierarchy
    iterable = ir.NameRef(extract_name(expr))
    reverse_coords = []
    while iterable in view_to_parent:
        reverse_coords.append(offset_names[iterable])
        iterable = view_to_parent[iterable]
    reverse_coords.reverse()
    view_coords = tuple(reverse_coords)
    # return base view with coordinate vector
    return iterable, view_coords


class UniformityChecker(StmtVisitor):
    """
    Check for non-uniform values/expressions.
    Must have non-uniform args known
    """
    pass


class LoopNestCheck(StmtVisitor):

    @contextmanager
    def scoped(self):
        self.mapped.append(dict())
        yield
        self.mapped.pop()

    def contains(self, ref):
        return any(ref in m for m in self.mapped)

    def __init__(self, syms):
        self.infer_type = ExprTypeInfer(syms)
        self.mapped = []

    def __call__(self, node):
        with self.scoped():
            return self.visit(node)

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Assign):
        target_type = self.infer_type(node.target)
        if isinstance(target_type, ir.ArrayType):
            if self.contains(node.target):
                pf = pretty_formatter()
                msg = f"Array references within a loop nest must be uniquely bound." \
                      f"Received duplicate entry for {pf(node.target)}."
                raise CompilerError(msg)

    @visit.register
    def _(self, node: ir.ForLoop):
        pass
