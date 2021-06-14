import itertools
import operator

from collections import deque, defaultdict
from functools import singledispatchmethod

import ir

from visitor import VisitorBase, walk_all


def extract_name(name):
    return name.name if isinstance(name, ir.NameRef) else name


class name_generator:
    def __init__(self, prefix):
        self.prefix = prefix
        self.gen = itertools.count()

    def make_name(self):
        return f"{self.prefix}_{next(self.gen)}"


class symbol_gen:

    def __init__(self, existing):
        self.names = existing
        self.added = set()
        self.gen = itertools.count()

    def __contains__(self, item):
        if isinstance(item, ir.NameRef):
            item = item.name
        return item in self.names

    def make_unique_name(self, prefix):
        name = f"{prefix}_{next(self.gen)}"
        while name in self.names:
            name = f"{prefix}_{next(self.gen)}"
        self.names.add(name)
        return name


def build_symbols(entry):
    # grabs all names that can be declared at outmermost scope
    names = set()
    if isinstance(entry, ir.Function):
        for arg in entry.args:
            names.add(extract_name(arg))
    for stmt in walk_all(entry):
        if isinstance(stmt, ir.Assign):
            if isinstance(stmt.target, ir.NameRef):
                names.add(extract_name(stmt.target))
        elif isinstance(stmt, ir.ForLoop):
            for target, _ in stmt.walk_assignments():
                names.add(extract_name(target))
    return names


def flatten_nested_branches(header: ir.IfElse):
    branches = [(header.test, header.if_branch)]
    stmt = header.else_branch
    while len(stmt) == 1 and isinstance(stmt[0], ir.IfElse):
        stmt = stmt[0]
        branches.append((stmt.test, stmt.if_branch))
        stmt = stmt.else_branch
    if stmt:
        # trailing non-empty else
        branches.append((None, stmt))
    return branches


def make_counter_based_bounds(iterables, arrays):
    # Make counters for access functions
    bounds = set()
    for iterable in iterables:
        if isinstance(iterable, ir.Counter):
            if iterable.stop is not None:
                bounds.add(iterable)
        elif isinstance(iterable, ir.Subscript):
            arr = arrays[iterable.value]
            leading_dim = arr.dims.elements[0]
            sl = iterable.slice
            if not isinstance(sl, ir.Slice):
                raise ValueError("non-slice subscript is not yet supported here for array iteration base")
            if sl.stop is None or leading_dim == sl.stop:
                bounds.add(ir.Counter(sl.start, leading_dim, sl.step))
            else:
                # Expand all bounds. This allows for more simpler redundancy checking than
                # ternary expressions
                bounds.add(ir.Counter(sl.start, sl.stop, sl.step))
                bounds.add(ir.Counter(sl.start, leading_dim, sl.step))
        else:
            arr = arrays[iterable]
            leading_dim = arr.dims.elements[0]
            bounds = (ir.Counter(ir.IntNode(0), leading_dim, ir.IntNode(1)))
    return bounds


def numeric_max_iter_count(bound):
    """
    Find max iteration count for an ir.Counter object with purely numeric parameters

    """
    assert(isinstance(bound.start, ir.IntNode))
    assert(isinstance(bound.stop, ir.IntNode))
    assert(isinstance(bound.step, ir.IntNode))
    start = bound.start.value
    stop = bound.stop.value
    step = bound.step.value
    if step == 0:
        raise ValueError
    elif stop <= start:
        if step > 0:
            return 0
        else:
            interval = start - stop
            step_mag = abs(step)
            count = interval / step_mag
            if interval % step_mag:
                count += 1
    else:
        if step < 0:
            # malformed expression, counts backwards away from stop
            raise ValueError
        interval = stop - start
        count = interval / step
        if interval % step:
            count += 1
    return count


def combine_numeric_bound_checks(bounds):
    numeric = set()
    for bound in bounds:
        if all(isinstance(param, ir.IntNode) for param in bound.subexprs):
            numeric.add(bound)
    if not numeric:
        return bounds
    updated = bounds.difference(numeric)
    numeric_iter = iter(numeric)
    min_bound = numeric_max_iter_count(next(numeric_iter))
    for bound in numeric_iter:
        min_bound = min(min_bound, numeric_max_iter_count(bound))
    updated.add(ir.IntNode(ir.IntNode(0), ir.IntNode(min_bound), ir.IntNode(1)))


def make_min_sliced_len_expr(slices, leading_dim_expr):
    if not slices:
        return leading_dim_expr
    s = slices[0]
    bound = leading_dim_expr if (s.stop is None or s.start == ir.IntNode(0)) else ir.BinOp(s.stop, s.start, "-")
    for s in slices[1:]:
        b = leading_dim_expr if (s.stop is None or s.start == ir.IntNode(0)) else ir.BinOp(s.stop, s.start, "-")
        bound = ir.IfExpr(ir.BinOp(bound, b, "<"), bound, b)
    return bound


def uniform_counter_params(counters):
    counters_iter = iter(counters)
    first_counter = next(counters_iter)
    first_start = first_counter.start
    first_top = first_bound.stop
    first_step = first_bound.step
    has_uniform_start = True
    has_uniform_stop = True
    has_uniform_step = True
    for counter in counters_iter:
        has_uniform_start &= (first_start == counter.start)
        has_uniform_stop &= (first_stop == counter.stop)
        has_uniform_step &= (first_step == counter.step)
    return has_uniform_start, has_uniform_stop, has_uniform_step


def make_loop_bound_exprs(header, syms, type):
    iterables = {it for (_, it) in header.walk_assignments()}
    if not iterables:
        raise ValueError
    # This leaves us with simple boundary conditions
    # We can use these to create the necessary access functions after grouping by start/step
    counters = make_counter_based_bounds(iterables)
    reduced_bounds = combine_numeric_bound_checks(counters)
    # optimize for the case where we have a single delinearized step size
    if len(reduced_bounds) == 1:
        # uniform parameters
        pass
    else:
        uniform_start, uniform_stop, uniform_step = uniform_counter_params(reduced_bounds)
        if uniform_start and uniform_step:
            # we can build a single induction bound by taking the min over stop at runtime
            pass
        elif uniform_stop and uniform_step:
            # we can build a single induction bound by taking the max over start at runtime
            pass
        else:
            pass

