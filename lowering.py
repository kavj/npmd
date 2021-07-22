import numbers
import operator
import typing

from collections import defaultdict
from functools import singledispatch, singledispatchmethod

import ir
import symbols

from TypeInterface import ArrayType
from visitor import VisitorBase, walk, walk_branches


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


def simplify_pow(base, coeff, in_place=False):
    if coeff.constant:
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
                if base.constant:
                    repl = wrap_constant(operator.pow(base.value, 2))
                else:
                    op = "*=" if in_place else "*"
                    repl = ir.BinOp(base, base, op)
                return repl
        elif isinstance(coeff, ir.FloatNode):
            if coeff.value == 0.5:
                if base.constant:
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


def is_innermost(header):
    return not any(stmt.is_loop_entry for stmt in walk_branches(header))


def unwrap_loop_body(node):
    return node.body if isinstance(node, (ir.ForLoop, ir.WhileLoop)) else node


def wrap_constant(value):
    if isinstance(value, ir.Constant):
        return value
    if isinstance(value, bool):
        return ir.BoolNode(value)
    elif isinstance(value, numbers.Integral):
        # needed for numpy compatibility
        return ir.IntNode(value)
    elif isinstance(value, numbers.Real):
        # needed for numpy compatibility
        return ir.FloatNode(value)
    else:
        msg = f"{value} of type {type(value)} is not recognized as a constant."
        raise TypeError(msg)


@singledispatch
def fold_if_numeric(expr):
    msg = f"fold expression not implemented for "
    raise NotImplementedError


@fold_if_numeric.register
def _(expr: ir.Constant):
    return expr


@fold_if_numeric.register
def _(expr: ir.NameRef):
    return expr


@fold_if_numeric.register
def _(expr: ir.Slice):
    start = fold_if_numeric(expr.start)
    stop = expr.stop
    if stop is not None:
        stop = fold_if_numeric(stop)
    step = fold_if_numeric(expr.step)
    slice_ = ir.Slice(start, stop, step)
    return slice_


@fold_if_numeric.register
def _(expr: ir.BinOp):
    left = fold_if_numeric(expr.left)
    right = fold_if_numeric(expr.right)
    if left.constant and right.constant:
        oper = binops[expr.op]
        res = oper(left.value, right.value)
        res = wrap_constant(res)
    else:
        res = ir.BinOp(left, right, expr.op)
    return res


@fold_if_numeric.register
def _(expr: ir.UnaryOp):
    value = try_fold_expression(expr.value)
    if value.constant:
        oper = unaryops[expr.op]
        repl = wrap_constant(oper(value))
    else:
        repl = ir.UnaryOp(value, expr.op)
    return repl


@fold_if_numeric.register
def _(expr: ir.Subscript):
    value = expr.value
    slice_ = fold_if_numeric(expr.slice)
    repl = ir.Subscript(value, slice_)
    return repl


def expand_inplace_op(expr):
    assert isinstance(expr, ir.BinOp)
    op_conversion = {"*=": "*", "-=": "-", "/=": "/", "//=": "//", "**=": "**", "|=": "|", "&=": "&",
                     "^=": "^", "~=": "~"}
    op = op_conversion[expr.op]
    return ir.BinOp(expr.left, expr.right, op)


def try_replace_unary_binary(expr):
    # Should fold anything like double negation first
    # and do subexpressions first
    if isinstance(expr, ir.UnaryOp):
        if expr.op == "-":
            expr = ir.BinOp(ir.IntNode(-1), expr.operand, "*")
    return expr


def discard_unbounded(iterables):
    bounded = {it for it in iterables if not (isinstance(it, ir.Counter) and it.stop is None)}
    return bounded


@singledispatch
def make_iter_counter(iterable, syms):
    msg = f"No method to make counter for {iterable}."
    raise NotImplementedError(msg)


@make_iter_counter.register
def _(iterable: ir.Subscript, syms):
    value = iterable.value
    array_type = syms.lookup(value)
    slice_ = fold_if_numeric(iterable.slice)
    if isinstance(slice_, ir.Slice):
        if slice_.stop is None:
            stop = array_type.dims[0]
        else:
            stop = ir.Min(leading_dim, slice_.stop)
        counter = ir.Counter(slice_.start, stop, slice_.step)
    else:
        # iterating over a single index subscript means iteration is bounded by
        # the second dimension.
        if array_type.ndims < 2:
            msg = f"Cannot iterate over a scalar reference {iterable}."
            raise ValueError(msg)
        counter = ir.Counter(ir.IntNode(0), array_type.dims[1], ir.Counter(1))
    return counter


@make_iter_counter.register
def _(iterable: ArrayType):
    return ir.Counter(ir.IntNode(0), iterable.dims[0], ir.IntNode(1))


@make_iter_counter.register
def _(iterable: ir.NameRef, syms):
    array_type = syms.lookup(iterable)
    counter = make_iter_counter(array_type)
    return counter


def make_loop_counters(iterables, syms):
    """
    Map a combination of array iterators and range and enumerate calls to a set of counters,
    which capture the appropriate intervals.

    In the case of arrays, this refers to the possibly non-linear
    index into the array at each step, which should be taken by the iterator at runtime.

    """
    # Make counters for access functions
    bounded = discard_unbounded(iterables)
    bounds = set()
    for iterable in bounded:
        if isinstance(iterable, ir.Subscript):
            arr = arrays[iterable.value]
            leading_dim = arr.dims.elements[0]
            sl = iterable.slice
            if not isinstance(sl, ir.Slice):
                raise ValueError("non-slice subscript is not yet supported here for array iteration base")
            if sl.stop is None or leading_dim == sl.stop:
                bounds.add(ir.Counter(sl.start, leading_dim, sl.step))
            else:
                # Expand all bounds, since we don't always know which is tighter.
                bounds.add(ir.Counter(sl.start, sl.stop, sl.step))
                bounds.add(ir.Counter(sl.start, leading_dim, sl.step))
        else:
            arr = arrays[iterable]
            leading_dim = arr.dims.elements[0]
            bounds.add(ir.Counter(ir.IntNode(0), leading_dim, ir.IntNode(1)))
    return bounds


def combine_numeric_checks(bounds: typing.List[ir.Counter]):
    numeric = {b for b in bounds if all(isinstance(param, ir.IntNode) for param in b.subexprs)}
    # Check for unbounded
    unbounded = {b for b in numeric if b.stop is None}
    numeric.difference_update(unbounded)
    if numeric:
        updated = set(bounds).difference(numeric)
        min_bound = min(numeric_max_iter_count(b) for b in numeric)
        updated.add(ir.Counter(ir.IntNode(0), ir.IntNode(min_bound), ir.IntNode(1)))
        bounds = updated
    return bounds


def make_min_sliced_len_expr(slices, leading_dim_expr):
    if not slices:
        return leading_dim_expr
    s = slices[0]
    bound = leading_dim_expr if (s.stop is None or s.start == ir.IntNode(0)) else ir.BinOp(s.stop, s.start, "-")
    for s in slices[1:]:
        b = leading_dim_expr if (s.stop is None or s.start == ir.IntNode(0)) else ir.BinOp(s.stop, s.start, "-")
        bound = ir.IfExpr(ir.BinOp(bound, b, "<"), bound, b)
    return bound


def make_explicit_iter_count(counter):
    if all(isinstance(subexpr, ir.IntNode) for subexpr in counter.subexprs):
        count = numeric_max_iter_count(counter)
    elif counter.step == ir.IntNode(1):
        count = ir.BinOp(counter.stop, counter.start, "-")
    else:
        on_false = ir.BinOp(counter.stop, counter.start, "-")
        # avoid integer division
        test = ir.BinOp(interval, counter.step, "&")
        on_true = ir.BinOp(on_false, ir.IntNode(1), "+")
        count = ir.IfExpr(test, on_true, on_false)
    return count


def merge_loop_counters(counters, syms, index_name):
    """
    Attempt to handle cases where we have relatively simple structured loop constraints.

    counters:
        A set of Counter functions, denoting affine sequences with imposed boundary conditions.
    syms:
        Symbol table for lookups of array parameters

    returns:        a loop index counter if a simple shared bound can be found without relying on explicit
        counter normalization of counters with symbolic parameters, otherwise None

    """

    # enumerate constructs lower to unbounded counters
    bounds = {c for c in counters if c.stop is not None}
    if len(bounds) == 1:
        counter, = bounds
        return counter

    diffs = defaultdict(set)
    for b in bounds:
        start = b.start
        stop = b.stop
        step = b.step
        if start.constant and stop.constant:
            pass
        diffs[step].add(ir.BinOp(b.stop, b.start, "-"))


    for step, intervals in diffs.items():
        for interval in intervals:

            pass
            # interval = try
        pass

    # for now, just declare this as min
    # for bound in bounds:
    #    if

    # optimize for the case where we have a single delinearized step size

    # Check if we can

    return counter


@singledispatch
def make_counter(base, syms):
    msg = f"Make counter not supported for input type {type(base)}"
    raise NotImplementedError(msg)


@make_counter.register
def _(base: ir.Counter, syms):
    return base


@make_counter.register
def _(base: ir.NameRef, syms):
    if not syms.is_array(base):
        msg = f"Variable {base} is iterated over without an array type declaration or assignment."
        raise KeyError(msg)
    sym = syms.lookup(base)
    if not sym.is_array:
        msg = f"Cannot create counter from non-array type {sym.type_}"
        raise TypeError(msg)
    arr = sym.type_
    leading = arr.dims[0]
    # this is delinearized, so not a direct access func
    counter = ir.Counter(ir.IntNode(0), leading, ir.IntNode(1))
    return counter


@make_counter.register
def _(base: ir.Subscript, syms):
    arr = syms.arrays[base.value]
    # avoid linearizing
    sl = base.slice
    if isinstance(sl, ir.Slice):
        start = wrap_constant(sl.start)
        stop = arr.dims[0]
        if sl.stop is not None:
            stop = ir.Min(stop, sl.stop)
        step = sl.step
        counter = ir.Counter(start, stop, step)
    else:
        # assume single subscript
        if len(arr.dims) < 2:
            raise ValueError
        start = ir.IntNode(0)
        stop = arr.dims[1]
        stop = wrap_constant(stop)
        step = ir.IntNode(1)
        counter = ir.Counter(start, stop, step)
    return counter


def make_loop_interval(targets, iterables, syms, loop_index):
    counters = []
    for target, iterable in zip(targets, iterables):
        c = make_counter(iterable, syms)
        counters.append(c)
    counter = merge_loop_counters(counters, syms, loop_index)
    return counters
