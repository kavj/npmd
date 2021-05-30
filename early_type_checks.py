import typing

from collections import defaultdict
from functools import singledispatchmethod

import ir

from visitor import VisitorBase


class TypeCanonicalizer:

    def __init__(self):
        self.types = {}

    def get_canonical_type(self, initial_type):
        """
        Placeholder
        This is meant to aggregate compatible types into single types in the case of aliasing and for cases
        where we merely approximate the original type.

        """
        t = self.types.get(initial_type)
        return t if t is not None else initial_type


def is_integer_only_op(op):
    return op in ("<<=", ">>=", "|=", "^=", "&=", "<<", ">>", "|", "^", "&")


def is_compare_op(op):
    return op in ("==", "!=", "<", "<=", ">", ">=")


def is_inplace_op(op):
    return op in ("+=", "-=", "*=", "/=", "//=", "%=", "%=", "**=", "<<=", ">>=", "|=", "^=", "&=", "@=")


class TypeErrors:
    def __init__(self):
        self.division = []
        self.unsigned = []
        self.unsafe_ops = []
        self.missing = []
        self.truth_testing = []
        self.integer_ops = []
        self.bad_array_refs = []

    def register_bad_divide(self, e):
        self.division.append(e)

    def register_bad_unsigned_op(self, e):
        self.unsigned.append(e)

    def register_missing(self, e):
        self.missing.append(e)

    def register_bad_truth(self, e):
        self.truth_testing.append(e)

    def register_bad_bit_op(self, e):
        self.integer_ops.append(e)

    def register_unsafe_op(self, e):
        self.unsafe_ops.append(e)

    def register_bad_array_reference(self, e):
        self.bad_array_refs.append(e)


type_checkable = typing.Optional[typing.Union[ir.ScalarType, ir.ArrayRef]]


class EarlyTypeVerifier(VisitorBase):

    def __call__(self, entry, type_info):
        self.entry = entry
        self.type_info = type_info
        self.required_casts = defaultdict(set)
        self.type_errors = TypeErrors()

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Function):
        if node is not self.entry:
            raise RuntimeError("Unsupported nested function")
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.Assign):
        target_type = self.visit(node.target)
        value_type = self.visit(node.value)
        if target_type is not None and value_type is not None:
            if target_type != value_type:
                self.required_casts[value_type].add(target_type)

    @visit.register
    def _(self, node: ir.NameRef) -> type_checkable:
        t = self.type_info.get(node)
        if t is None:
            self.type_errors.register_missing(node)
            return

    @visit.register
    def _(self, node: ir.IfElse):
        test_type = self.visit(node.test)
        if test_type is None or isinstance(test_type, ir.ArrayRef):
            self.type_errors.register_bad_truth(node.test)
        self.visit(node.if_branch)
        self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.Slice) -> typing.Optional[ir.Slice]:
        for subexpr in node.subexprs:
            subtype = self.visit(subexpr)
            if not isinstance(subtype, ir.ScalarType):
                return
            if not subtype.integral:
                return
        return node

    @visit.register
    def _(self, node: ir.Subscript) -> type_checkable:
        slice_type = self.visit(node.slice)
        base_type = self.visit(node.value)
        output_type = None
        if slice_type is None or not isinstance(base_type, ir.ArrayRef):
            return output_type
        reduce_dims_by = 1 if isinstance(slice_type, ir.ScalarType) else 0
        output_ndims = base_type.ndims - reduce_dims_by
        if output_ndims < 0:
            self.type_errors.register_bad_array_reference(node)
        elif output_ndims == 0:
            output_type = ir.ArrayRef.dtype
        else:
            output_type = ir.ArrayRef(base_type.dtype, output_ndims)
        return output_type

    @visit.register
    def _(self, node: ir.IfExpr):
        test_type = self.visit(node.test)
        if test_type is None:
            return
        elif isinstance(test_type, ir.ArrayRef):
            # we can still check whether we would get a valid output type
            # given an unambiguous truth testable type
            self.type_errors.register_bad_truth(test_type)
        if_type = self.visit(node.if_expr)
        else_type = self.visit(node.else_expr)
        return if_type if if_type == else_type is not None else None

    @visit.register
    def _(self, node: ir.ForLoop):
        for target, iterable in node.walk_assignments():
            iterable_type = self.visit(iterable)
            if not isinstance(iterable_type, ir.ArrayRef):
                self.type_errors.register_bad_array_reference(iterable_type)
            if iterable_type.ndims == 1:
                predicted_target_type = iterable_type.dtype
            else:
                predicted_target_type = ir.ArrayRef(iterable_type.dtype, iterable_type.ndims - 1)
            target_type = self.visit(target)
            if target_type is not None:
                if target_type != predicted_target_type:
                    self.required_casts[predicted_target_type].add(target_type)

    @visit.register
    def _(self, node: ir.WhileLoop):
        test_type = self.visit(node.test)
        if test_type is None or isinstance(test_type, ir.ArrayRef):
            self.type_errors.register_bad_truth(node.test)
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.UnaryOp) -> type_checkable:
        operand_type = self.visit(node.operand)
        if operand_type is None:
            return
        dtype = operand_type.dtype if isinstance(operand_type, ir.ArrayRef) else operand_type
        if node.op == "-":
            if not dtype.signed:
                self.type_errors.register_bad_unsigned_op(node)
                return  # can't check further, terminate checks
        elif node.op in ("~", "not"):
            if not dtype.integral:
                self.type_errors.register_bad_bit_op(node)
        return operand_type

    @visit.register
    def _(self, node: ir.BinOp) -> type_checkable:
        op = node.op
        left_type = self.visit(node.left)
        right_type = self.visit(node.right)
        output_type = None
        if left_type is not None and right_type is not None:
            left_dtype = left_type.dtype if isinstance(left_type, ir.ArrayRef) else left_type
            right_dtype = right_type.dtype if isinstance(right_type, ir.ArrayRef) else right_type
            if op == "/=":
                if not left_dtype.integral:
                    self.type_errors.register_bad_divide(node)
            elif is_integer_only_op(op):
                if not left_dtype.integral or right_dtype.integral:
                    self.type_errors.register_bad_bit_op(node)
            elif left_dtype.boolean or right_dtype.boolean:
                self.type_errors.register_unsafe_op(node)
            # we should have something reasonable
            # first we check if this needs an array type output
            if is_compare_op(op):
                # Todo: need an actual lookup table for bitwidths..
                output_dtype = ir.ScalarType(signed=True, boolean=True, integral=True, bitwidth=8)
            else:
                if left_dtype == right_dtype or is_inplace_op(op):
                    output_dtype = left_dtype
                elif left_dtype.integral != right_dtype.integral:
                    output_dtype = ir.ScalarType(signed=True, boolean=False, integral=False, bitwidth=64)
                else:
                    output_dtype = left_dtype if left_dtype.bitwidth >= right_dtype.bitwidth else right_dtype
            if output_dtype is not None:
                if isinstance(left_type, ir.ArrayRef):
                    if isinstance(right_type, ir.ArrayRef):
                        # Array op Array (no handling of broadcasting in this api)
                        if left_type.ndims == right_type.ndims:
                            output_type = ir.ArrayRef(output_dtype, left_type.ndims)
                    else:  # Array op scalar
                        output_type = ir.ArrayRef(output_dtype, left_type.ndims)
                elif isinstance(right_type, ir.ArrayRef):
                    # Scalar op array
                    output_type = ir.ArrayRef(output_dtype, right_type.ndims)
        return output_type
