import itertools
import operator

from collections import deque, defaultdict
from contextlib import ContextDecorator
from functools import singledispatch, singledispatchmethod

import ir

from used_by import reads_writes
from value_numbering import branch_value_numbering
from visitor import VisitorBase, walk_all, walk_branches


def extract_name(name):
    return name.name if isinstance(name, ir.NameRef) else name

def is_innermost(header):
    return not any(stmt.is_loop_entry for stmt in walk_branches(header))


def unwrap_loop_body(node):
    return node.body if isinstance(node, (ir.ForLoop, ir.WhileLoop)) else node


def block_partition(stmts):
    partitions = []
    curr = []
    for stmt in stmts:
        if isinstance(stmt, (ir.IfElse, ir.CascadeIf, ir.ForLoop, ir.WhileLoop)):
            if curr:
                partitions.append(curr)
                curr = []
            partitions.append(stmt)
    if curr:
        partitions.append(curr)
    return tuple(partitions)


def constraint_variables(node: ir.ForLoop):
    iterable_constraints = set()
    range_constraints = set()
    for _, iterable in node.walk_assignments():
        if isinstance(iterable, ir.Counter):
            if iterable.stop is not None:
                #  keep step since we may not know sign
                range_constraints.add(iterable)
        else:
            iterable_constraints.add(iterable)
    return iterable_constraints, range_constraints


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


class DeclBuilder(VisitorBase):
    """
    Indicates where to place declarations when lowering

    """

    def __call__(self, entry):
        self.decls = None
        self.cumulative = None
        self.visit(entry)
        decls = self.decls
        self.decls = self.cumulative = None
        return decls

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Function):
        self.decls = defaultdict(set)
        args = {arg for arg in node.args}
        self.decls[id(node)] = args
        self.cumulative = args.copy()
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.IfElse):
        stashed = self.cumulative
        self.cumulative = self.cumulative.copy()
        if_branch = self.visit(node.if_branch)
        self.cumulative = stashed.copy()
        else_branch = self.visit(node.else_branch)
        if if_branch and else_branch:
            shared = if_branch.intersection(else_branch)
            if_branch.difference_update(shared)
            else_branch.difference_update(shared)
            return shared
        else:
            return ()

    @visit.register
    def _(self, node: list):
        for stmt in node:
            decls = self.visit(stmt)
            self.cumulative.update(decls)
            self.decls[id(node)].update(decls)

    @visit.register
    def _(self, node: ir.ForLoop):
        stashed = self.cumulative
        self.cumulative = self.cumulative.copy()
        for_id = id(node)
        body_id = id(node.body)
        if(len(node.assigns) != 1):
            # more trouble than it's worth
            raise ValueError("unable to build loop declarations for a multi-variate header")
        target, _ = next(node.walk_assignments())
        # this should be scoped to a single loop
        assert target not in self.cumulative
        self.decls[for_id].update(target)
        self.cumulative.add(target)
        self.visit(node.body)
        self.cumulative = stashed

    @visit.register
    def _(self, node: ir.WhileLoop):
        stashed = self.cumulative
        self.cumulative = self.cumulative.copy()
        self.visit(node.body)
        self.cumulative = stashed

    @visit.register
    def _(self, node: ir.Assign):
        target = node.target
        if isinstance(target, ir.NameRef):
            if target not in self.cumulative:
                self.cumulative.add(target)
                return (target,) if isinstance(target, ir.NameRef) else ()


def extract_elifs(header: ir.IfElse):
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


def flatten_if_branch_tests(header: ir.IfElse):
    """
    This flattens perfectly checks into a long single conditional without reordering.

    """

    body = header.if_branch
    tests = [header.test]
    while ((len(current) == 1)
            and isinstance(body[0], ir.IfElse)
            and not body[0].else_branch):
        tests.append(body[0]).test
        body = body[0].if_branch
    if len(tests) > 1:
        header = ir.IfElse(ir.BoolOp(tuple(tests), "and"), body, header.else_branch, header.pos)
    return header


def cascade_flatten_branch_tests(header: ir.IfElse):
    header = flatten_if_branch_tests(header)
    repl_if_body = []
    for stmt in header.if_branch:
        if isinstance(stmt, ir.IfElse):
            stmt = flatten_if_branch_tests(stmt)
        repl_if_body.append(stmt)
    repl_else_body = []
    for stmt in header.else_branch:
        if isinstance(stmt, ir.IfElse):
            stmt = flatten_if_branch_tests(stmt)
        repl_else_body.append(stmt)
    header.if_branch = repl_if_body
    header.else_branch = repl_else_body
    return header


def renest_branches(header: ir.IfElse, varying):
    """
    group cascaded branches, based on whether they are uniformly taken

    """
    branches = extract_elifs(header)
    if len(branches) < 3:
        return header
    else_branch = branches.pop()
    lead = branches.pop()
    lead_is_uniform = lead.test not in varying
    if_branches = [lead.test]
    tests = [lead.if_branch]
    for test, body in reversed(branches):
        is_uniform = test not in varying
        if is_uniform != lead_is_uniform:
            if_branches.reverse()
            tests.reverse()
            combined = ir.CascadeIf(tests, if_branches, else_branch)
            if_branches = []
            tests = []
            else_branch = combined
            lead_is_uniform = is_uniform
        if_branches.append(body)
        tests.append(test)
    if_branches.reverse()
    tests.reverse()
    combined = ir.CascadeIf(tests, if_branches, else_branch)
    return combined


def has_break_stmt(entry):
    if isinstance(entry, (ir.ForLoop, ir.WhileLoop)):
        entry = entry.body
    return any(isinstance(stmt, (ir.Break, ir.Return)) for stmt in entry)


def contains_any_break(entry):
    for stmt in walk_branches(entry):
        if isinstance(stmt, ir.Break):
            return True


def contains_varying_break(entry, uniform):
    # check for false loop conversions before this
    # This should run before flattening of nested branches.
    for stmt in entry:
        if isinstance(stmt, ir.IfElse):
            if stmt.test in uniform:
                if contains_varying_break(stmt.if_branch, uniform) or contains_varying_break(stmt.else_branch):
                    return True
            else:
                # Break is only varying if it appears along a varying path. Otherwise any remaining lanes
                # exit the loop. For elif type statements, we still have to consider that only some may exit.
                if contains_any_break(stmt.if_branch) or contains_any_break(stmt.else_branch):
                    return True
    return False


def loop_body_may_exit_func(entry):
    if isinstance(entry, (ir.ForLoop, ir.WhileLoop)):
        entry = entry.body
    for stmt in walk_all(entry):
        if isinstance(stmt, ir.Return):
            return True
    return False


def can_convert_break_to_noop(entry):
    if isinstance(entry, (ir.ForLoop, ir.WhileLoop)):
        entry = entry.body
    for stmt in walk_branches(entry):
        if isinstance(stmt, (ir.ForLoop, ir.WhileLoop)):
            continue
        if isinstance(ir.Assign):
            if isinstance(stmt.value, (ir.BinOp, ir.UnaryOp)):
                if isinstance(stmts.target, ir.Subscript):
                    # If this is marked in-place, or the same subscript appears as a sub-expression
                    # of the right hand side, and the left hand side refers to a varying array reference,
                    # then we can convert false predicates to identity ops.
                    pass
            else:
                return False


def expand_inplace_op(expr):
    assert isinstance(expr, ir.BinOp)
    op_conversion = {"*=": "*", "-=": "-", "/=" : "/", "//=": "//", "**=": "**", "|=": "|", "&=" : "&",
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


def make_noop(expr, predicate):
    if isinstance(expr, ir.BinOp):
        if expr.in_place:
            expr = expand_inplace_op(expr)

    elif isinstance(expr, ir.UnaryOp):
        op = expr.op


def make_constant_like(expr, types, value):
    t = types[expr]
    if t == int:  # Todo: this won't capture all, maybe an issubclass?
        value = ir.IntNode(value)
    elif t == float:
        value = ir.FloatNode(value)
    else:
        raise TypeError


def try_convert_varying_to_noop(stmts, symbols, predicate, types):
    repl = []

    for stmt in stmts:
        if isinstance(stmt, ir.Assign):
            # convertible if inplace arithmetic or unary op
            # or if not explicitly in place but the target is both read and written by the assignment
            lhs = stmt.target
            if lhs == stmt.value:
                # noop
                continue
            if stmt.in_place:
                # look up conversion, overwrite statement
                expr = expand_inplace_op(stmt.value)
            else:
                expr = stmt.value
            if isinstance(expr, ir.UnaryOp):
                expr = try_replace_unary_binary(expr)
            if isinstance(expr, ir.BinOp):
                # primary practical case of interest
                # If we support non-commutative operands later,
                # this must be revisited
                if expr.left == lhs:
                    rhs = expr.right
                elif expr.right == lhs:
                    rhs = expr.left
                else:
                    # can't simplify
                    repl.append(stmt)
                    continue
                # These typically lower to bitwise operations, but those aren't defined for floating
                # point arithmetic. They're typically just an extension of simd ops. This still communicates
                # an unambiguous intent to later stages.
                zero = make_constant_like(expr.right, types, 0)
                one = make_constant_like(expr.right, types, 1)
                if op == "+":
                    expr = ir.BinOp(expr.left, ir.IfExpr(predicate, expr.right, zero), "+")
                elif op == "*":
                    expr = ir.BinOp(expr.left, ir.IfExpr(predicate, expr.right, one), "*")
                elif op == "/":
                    expr = ir.BinOp(expr.left, ir.IfExpr(predicate, expr.right, one), "/")
                elif op == "//":
                    expr = ir.BinOp(expr.left, ir.IfExpr(predicate, expr.right, one), "//")
                # ignore other cases
                if stmt.in_place:
                    expr = ir.BinOp(expr.lhs, expr.rhs, f"{expr.op}=")
                stmt = ir.Assign(stmt.target, expr, stmt.pos)
                repl.append(stmt)
    return repl


def make_loop_counters(iterables, arrays):
    """
    Map a combination of array iterators and range and enumerate calls to a set of counters,
    which capture the appropriate intervals.

    In the case of arrays, this refers to the possibly non-linear
    index into the array at each step, which should be taken by the iterator at runtime.

    """
    # Make counters for access functions
    bounds = defaultdict(set)
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
                # Expand all bounds, since we don't always know which is tighter.
                bounds[ir.Counter(sl.start, sl.stop, sl.step)].add(iterable)
                bounds[ir.Counter(sl.start, leading_dim, sl.step)].add(iterable)
        else:
            arr = arrays[iterable]
            leading_dim = arr.dims.elements[0]
            bounds[ir.Counter(ir.IntNode(0), leading_dim, ir.IntNode(1))] = set(*iterables)
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


def combine_numeric_checks(bounds):
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


def make_max_expr(exprs):
    params = set()
    constants = set()
    for expr in exprs:
        if expr.constant:
            constants.add(expr)
        else:
            params.add(expr)
    if constants:
        assert all(isinstance(c, ir.IntNode) for c in constants)
        max_value = max(c.value for c in constants)
        params.add(ir.IntNode(c))
    return ir.Max(tuple(params))


def make_min_expr(exprs):
    if not exprs:
        raise ValueError
    params = set()
    constants = set()
    for expr in exprs:
        if expr.constant:
            constants.add(expr)
        else:
            params.add(expr)
    if constants:
        assert all(isinstance(c, ir.IntNode) for c in constants)
        min_value = min(c.value for c in constants)
        params.add(ir.IntNode(c))
    if len(params) == 1:
        min_expr = params.pop()
    else:
        min_expr = ir.Min(tuple(params))
    return min_expr


def make_max_expr(exprs):
    if not exprs:
        raise ValueError
    params = set()
    constants = set()
    for expr in exprs:
        if expr.constant:
            constants.add(expr)
        else:
            params.add(expr)
    if constants:
        assert all(isinstance(c, ir.IntNode) for c in constants)
        max_value = max(c.value for c in constants)
        params.add(ir.IntNode(c))
    if len(params) == 1:
        max_expr = params.pop()
    else:
        max_expr = ir.Max(tuple(params))
    return max_expr


def uniform_counter_params(counters):
    if not counters:
        raise ValueError("No counters were provided")
    counters_iter = iter(counters)
    first_counter = next(counters_iter)
    first_start = first_counter.start
    first_top = first_bound.stop
    first_step = first_bound.step
    uniform_start = True
    uniform_stop = True
    uniform_step = True
    for counter in counters_iter:
        uniform_start &= (first_start == counter.start)
        uniform_stop &= (first_stop == counter.stop)
        uniform_step &= (first_step == counter.step)
    return uniform_start, uniform_stop, uniform_step


def group_by_start_stop(counters):
    grouped = defaultdict(set)
    for counter in counters:
        grouped[(counter.start, counter.stop)].add(counter)
    return grouped


def group_by_start_step(counters):
    grouped = defaultdict(set)
    for counter in counters:
        grouped[(counter.start, counter.step)].add(counter)
    return grouped


def group_by_stop_step(counters):
    grouped = defaultdict(set)
    for counter in counters:
        grouped[(counter.stop, counter.step)].add(counter)
    return grouped


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


def make_loop_interval(counters, syms, index_name):
    """
    Attempt to handle cases where we have relatively simple structured loop constraints.

    counters:
        A set of Counter functions, denoting affine sequences with imposed boundary conditions.
    syms:
        Symbol table for lookups of array parameters

    returns:
        a loop index counter if a simple shared bound can be found without relying on explicit
        counter normalization of counters with symbolic parameters, otherwise None

    """

    reduced_bounds = combine_numeric_bound_checks(counters)
    # optimize for the case where we have a single delinearized step size
    if len(reduced_bounds) == 1:
        # uniform parameters
        intervals = reduced_bounds
    else:
        by_start_stop = group_by_start_stop(reduced_bounds)
        by_start_step = group_by_start_step(reduced_bounds)
        by_stop_step = group_by_stop_step(reduced_bounds)

        start_stop_count = len(by_start_stop)
        start_step_count = len(by_start_step)
        stop_step_count = len(by_stop_step)
        min_count = min(start_stop_count, start_step_count, stop_step_count)
        # Here we're optimizing based on step > 0. Counters pertaining to negative
        # items should be converted for simplicity. It's not possible to do these things
        # efficiently if any array could be iterate forward or backwards on any call.
        if start_stop_count == min_count:
            intervals = []
            for (start, stop), group in by_start_stop.items():
                step = make_max_expr((g.step for g in group))
                intervals.append(ir.Counter(start, stop, step))
        elif start_step_count == min_count:
            intervals = []
            for (start, step), group in by_start_step.items():
                stop = make_min_expr((g.stop for g in group))
                intervals.append(ir.Counter(start, stop, step))
        elif stop_step_count == min_count:
            intervals = []
            for (stop, step), group in by_stop_step.items():
                start = make_max_expr((g.start for g in group))
                intervals.append(ir.Counter(start, stop, step))
        iter_counts = []
        for interval in intervals:
            on_false = ir.BinOp(interval.stop, interval.start, "-")
            # avoid cost of integer division
            test = ir.BinOp(on_false, interval.step, "&")
            on_true = ir.BinOp(on_false, ir.IntNode(1), "+")
            iter_counts.append(ir.IfExpr(test, on_true, on_false))
        if len(iter_counts) == 1:
            counter = iter_counts.pop()
        else:
            counter = ir.Counter(ir.IntNode(0), ir.Min(tuple(it for it in iter_counts)), ir.IntNode(1))

    return counter


def simplify_for_loop(header, symbols, types, arrays, offsets):
    # Todo: not sure whether this should be folded into code gen
    iterables = {iterable for (_, iterable) in header.walk_assignments()}
    counters = make_loop_counters(iterables, symbols)
    bound = make_loop_interval(counters, symbols)
    loop_index = symbols.make_unique_name()
    by_start_step = group_by_start_step(bounds)
    stmts = []
    for (start, step), group in by_start_step.items():
        for target, iterable in group:
            if isinstance(iterable, ir.Counter):
                if step != ir.IntNode(1):
                    expr = ir.BinOp(loop_index, step, "*")
                    if start != ir.IntNode(0):
                        expr = ir.BinOp(expr, start, "+")
                    assign = ir.Assign(target, expr, header.pos)
                elif start != ir.IntNode(0):
                    assign = ir.Assign(target, ir.BinOp(loop_index, start, "+"), header.pos)
                else:
                    assign = ir.Assign(target, loop_index, header.pos)
                stmts.append(assign)
            else:
                arr = arrays[extract_name(iterable)]
                base = arr.base
                offset_stride = offsets.get(arr)
                leading_dim = arr.dims[0]
                if arr.ndims == 1:
                    if isinstance(target, ir.ArrayRef):
                        raise TypeError
                    if step != ir.IntNode(1):
                        s = ir.BinOp(loop_index, step, "*")
                        if start != ir.IntNode(0):
                            s = ir.BinOp(s, start, "+")
                    elif start != ir.IntNode(0):
                        s = ir.BinOp(loop_index, start, "+")
                    else:
                        s = loop_index
                    if offset_expr is not None:
                        s = ir.BinOp(offset_expr, s)
                    # not completely correct
                    value = ir.BinOp(ir.BinOp(s, offset_expr, "+"))
                    assign = ir.Assign(target, value)
                else:
                    # need to store the corresponding offset
                    pass
    stmts.extend(header.body)
    loop_header = ir.ForLoop([(loop_index, bound)], stmts, header.pos)
    return loop_header


# refactoring of branch invariant assigned expressions

def find_subscripted_clobber_seqs():
    pass


def find_common_prefix():
    pass


def partition_branches(node):
    if isinstance(node, ir.IfElse):
        partitions =  (block_partition(node.if_branch), block_partition(node.else_branch))
    elif isinstance(node, ir.CascadeIf):
        partitions = [block_partition(branch) for branch in node.if_branches]
        partitions.append(block_partition(node.else_branch))
        partitions = tuple(partitions)
    else:
        raise TypeError("not a valid branch node")
    return partitions


def find_branch_assign_stats(node):
    tracking, name_assigns, subscript_assigns = branch_value_numbering(node)
    partitioned = partition_branches(node)
    if not assigned[-1] and not subscript_assigns[-1]:
        # no assignments in else branch
        # This may need to expand later in case of subsequent calls
        assigned.pop()
        subscript_assigns.pop()
    common_name_targets = set(assigned[0].keys())
    common_subscripts = set(subscript_assigns.keys())
    partial_targets = common_name_targets.copy()
    partial_subscripts = common_subscripts.copy()
    for assign, sassign in zip(assigned[1:], subscript_assigns[1:]):
        keys = assign.keys()
        skeys = sassign.keys()
        common_name_targets.intersection_update(keys)
        common_subscripts.intersection_update(skeys)
        partial_targets.update(keys)
        partial_subscripts.update(skeys)

    return (tracking, name_assigns, subscript_assigns, common_name_targets,
            common_subscripts, partial_targets, partial_subscripts)

def refactor_branch(node, ctx):


    (tracking, name_assigns, subscript_assigns, common_name_targets,
     common_subscripts, partial_targets, partial_subscripts) = find_branch_assign_stats(node)
    # F

    # Here we're tracking value numbering from a common branch point
    # to check for either assigned values or target access functions, which share
    # the same registered value number across 2 or more branches. As long as these
    # do not
