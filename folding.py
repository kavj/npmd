import math
import numbers
import operator
from collections import defaultdict
from functools import singledispatchmethod

import numpy as np

import ir
from visitor import walk_all, TransformBase

unaryops = {"+": operator.pos,
            "-": operator.neg,
            "~": operator.inv,
            "not": operator.not_,
            }

binops = {"+": operator.add,
          "-": operator.sub,
          "*": operator.mul,
          "/": operator.truediv,
          "//": operator.floordiv,
          "%": operator.mod,
          "**": operator.pow,
          "@": operator.matmul,
          "+=": operator.iadd,
          "-=": operator.isub,
          "*=": operator.imul,
          "/=": operator.ifloordiv,
          "//=": operator.itruediv,
          "%=": operator.imod,
          "**=": operator.ipow,
          "@=": operator.imatmul,
          "==": operator.eq,
          "!=": operator.ne,
          "<": operator.lt,
          "<=": operator.le,
          ">": operator.gt,
          ">=": operator.ge,
          "<<": operator.lshift,
          ">>": operator.rshift,
          "&": operator.and_,
          "|": operator.or_,
          "^": operator.xor,
          "<<=": operator.ilshift,
          ">>=": operator.irshift,
          "&=": operator.iand,
          "|=": operator.ior,
          "^=": operator.ixor,
          "isnot": NotImplemented,
          "in": NotImplemented,
          "notin": NotImplemented
          }

bitwise_binops = {"<<", ">>", "&", "|", "^", "<<=", ">>=", "&=", "|=", "^="}

foldable_cast_types = {bool, int, float, np.bool_, np.int32, np.int64, np.float32, np.float64}


def wrap_constant(c):
    if isinstance(c, bool):
        return ir.BoolNode(c)
    elif isinstance(c, numbers.Integral):
        return ir.IntNode(c)
    elif isinstance(c, numbers.Real):
        return ir.FloatNode(c)
    else:
        msg = f"Can't construct constant node for unsupported constant type {type(c)}"
        raise NotImplementedError(msg)


def simplify_pow(base, coeff, in_place=False):
    if isinstance(coeff, ir.Constant):
        if not isinstance(coeff, (ir.IntNode, ir.FloatNode)):
            msg = f"Cannot evaluate pow operation with power of type {type(coeff)}"
            raise RuntimeError(msg)
        if isinstance(coeff, ir.IntNode):
            coeff = coeff.value
            if coeff == 0:
                return ir.IntNode(1)
            elif coeff == 1:
                return base
            elif coeff == 2:
                if isinstance(base, ir.Constant):
                    repl = wrap_constant(operator.pow(base.value, 2))
                else:
                    op = "*=" if in_place else "*"
                    repl = ir.BinOp(base, base, op)
                return repl
        elif isinstance(coeff, ir.FloatNode):
            if coeff.value == 0.5:
                if isinstance(base, ir.Constant):
                    left = base.value
                    try:
                        left = math.sqrt(left)
                        return wrap_constant(left)
                    except (TypeError, ValueError):
                        msg = f"The source code may compute a square root of {base.value}" \
                              f"at runtime, which is invalid. We cautiously refuse to compile this."
                        raise RuntimeError(msg)
                else:
                    return ir.Call("sqrt", args=(base,), keywords=())
    return ir.BinOp(base, coeff, "**=" if in_place else "**")


def simplify_binop(left, right, op):
    if right.constant:
        if left.constant:
            const_folder = binops.get(op)
            if op in bitwise_binops and not isinstance(left, ir.IntNode):
                raise RuntimeError(f"Source code contains an invalid bit field expression"
                                   f"{op} requires integral operands, received {left}"
                                   f"and {right}")
            try:
                value = const_folder(left.value, right.value)
                return wrap_constant(value)
            except (TypeError, ValueError):
                raise RuntimeError(f"Extracted source expression {left} {op} {right}"
                                   f"cannot be safely evaluated.")
        else:
            if op in ("**", "*=") and right.value in (0, 1, 2, 0.5):
                return simplify_pow(left, right, op == "**=")
        return ir.BinOp(left, right, op)


class FoldExpressions(TransformBase):
    """
    Folds expressions with constant evaluations, returning an updated map of new expressions.
    This can be used to selectively inline variables with a single constant definition, in cases where
    this definition can be proven to reach every use that may be executed.

    This is considerably simpler to prove here than it would be in arbitrary Python, due to a lack of arbitrary
    try-catch blocks and a number of compile time restrictions to reduce the required runtime checks.

    """

    def __call__(self, entry, named_constants):
        self.named_constants = named_constants
        func = self.visit(entry)
        self.named_constants = None
        return func

    def as_constant(self, node):
        return self.named_constants.get(node, node)

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op
        const_left = self.as_constant(left)
        const_right = self.as_constant(right)
        if const_right.constant:
            repl = simplify_binop(const_left, const_right, op)
            if repl.constant or not isinstance(repl, ir.BinOp):
                return repl
        return node

    @visit.register
    def _(self, node: ir.BoolOp):
        operands = []
        for operand in node.operands:
            repl = self.visit(operand)
            operands.append(repl)
        early_return_on = False if node.op == "and" else True
        repl_operands = []
        for operand in node.operands:
            operand = self.visit(operand)
            operand = self.as_constant(operand)
            if isinstance(operand, ir.Constant):
                if operator.truth(operand) == early_return_on:
                    return ir.BoolNode(early_return_on)
            else:
                repl_operands.append(operand)
        if not repl_operands:
            # all operands had constant truth tests and none satisfied
            # early return criteria
            return ir.BoolNode(True) if node.op == "and" else ir.BoolNode(False)
        else:
            return ir.BoolOp(tuple(repl_operands), node.op)

    @visit.register
    def _(self, node: ir.Call):
        args = tuple(self.visit(arg) for arg in node.args)
        keywords = tuple((kw, self.visit(value)) for (kw, value) in node.keywords)
        return ir.Call(node.funcname, args, keywords)

    @visit.register
    def _(self, node: ir.IfExpr):
        test = self.as_constant(self.visit(node.test))
        if test.constant:
            if operator.truth(test):
                repl = self.visit(node.if_expr)
            else:
                repl = self.visit(node.else_expr)
        else:
            repl = ir.IfExpr(test, self.visit(node.if_expr), self.visit(node.else_expr))
        return repl

    @visit.register
    def _(self, node: ir.Reversed):
        if isinstance(node, ir.Counter):
            # check if range
            if node.stop is None:
                raise RuntimeError("Cannot reverse enumerate")
            repl = []
            for key, s in (node.start, node.stop, node.step):
                s = self.visit(s)
                const_s = self.as_constant(s)
                if not isinstance(s, ir.NameRef):
                    s = const_s
                repl.append(s)
            # swap start and stop
            stop, start, step = repl

            # negate step
            if isinstance(step, ir.IntNode):
                step = ir.IntNode(operator.neg(step.value))
            else:
                # try to fold negation
                step = self.visit(ir.UnaryOp(step, "-"))

            return ir.Counter(start, stop, step)

        return ir.Reversed(self.visit(node.iterable))

    @visit.register
    def _(self, node: ir.UnaryOp):
        operand = self.visit(node.operand)
        if isinstance(operand, ir.UnaryOp):
            # fold double negate
            if node.op == operand.op == "-":
                node = node.operand.operand
        as_const = self.as_constant(operand)
        if as_const is not operand:
            if node.op == "~":
                if isinstance(as_const, ir.Constant) and not isinstance(as_const, ir.IntNode):
                    msg = f"Cannot apply unary inversion to a non-integer operand, received {type(as_const)}"
                    raise RuntimeError(msg)
            dispatcher = unaryops.get(node.op)
            try:
                result = dispatcher(operand.value)
                return wrap_constant(result)
            except (TypeError, ValueError):
                msg = f"The source code may evaluate an invalid or unsupported constant expression " \
                      f"{node.op} {operand} at runtime. Since this performs limited runtime error" \
                      f"checking and this is almost certainly an error, we are halting here."
                raise RuntimeError(msg)
        return ir.UnaryOp(operand, node.op)

    @visit.register
    def _(self, node: ir.Cast):
        expr = self.visit(node.expr)
        as_const = self.as_constant(expr)
        repl = ir.Cast(self.visit(node.expr), node.as_type)
        if isinstance(as_const, ir.Constant):
            if node.as_type in foldable_cast_types:
                method = getattr(node.as_type, "__call__")
                value = method(as_const.value)
                if isinstance(value, bool):
                    repl = ir.BoolNode(value)
                elif isinstance(value, numbers.Integral):
                    repl = ir.IntNode(value)
                elif isinstance(value, numbers.Real):
                    repl = ir.FloatNode(value)
        return repl


def collect_assignments(node):
    by_target = defaultdict(list)
    if isinstance(node, ir.Assign):
        by_target[node.target].append(node)
    elif isinstance(node, ir.Walkable):
        for stmt in walk_all(node):
            if isinstance(stmt, ir.Assign):
                by_target[stmt.target].append(stmt)
    return by_target


def fold_constant_expressions(entry):
    assigns = collect_assignments(entry)
    constants = {}
    for target, stmts in assigns.items():
        if len(stmts) == 1:
            value = stmts[0].value
            if value.constant:
                constants[target] = value
    fold_exprs = FoldExpressions()
    func = fold_exprs(entry, constants)
    return func
