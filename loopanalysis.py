import ir
from reachingcheck import ReachingCheck
from visitor import walk_branches

"""
All utilities used
"""


# function utilities


def starts_from_zero(ir_counter):
    return ir_counter.start == ir.IntNode(0)


def has_unit_step(ir_counter):
    return ir_counter.step == ir.IntNode(1)


def suitable_loop_counters(header: ir.ForLoop, clobbers, liveouts):
    counters = set()
    for target, value in header.walk_assignments():
        if isinstance(value, ir.Counter):
            if target not in clobbers and target not in liveouts:
                if starts_from_zero(value) and has_unit_step(value):
                    counters.add(target)
    return counters


def find_loop_counter_bound(header, array_refs, loop_index):
    # find expression for counter limits
    # This needs a way to deal with subscripted array references
    ub = set()
    for target, value in header.walk_assignments():
        if target is loop_index:
            continue
        # any array refs should eventually get a dimref here
        if isinstance(value, ir.Counter):
            # avoid intermingling numeric and symbolic components, since we can evaluate
            # min integer at compile time
            if isinstance(value.stop, ir.IntNode):
                ub.add(value.stop.value)
            elif value.stop is not None:
                ub.add(value.stop)
        else:
            ub.add(value)
    # need to collapse any numeric results, then return a min expression
    return ub


def lower_loop_iterator(header, clobbers, liveouts, array_refs):
    """
    convert a for or while loop construct into something suitable for widening
    don't worry about widening or sinking here.
    live out variables must be identified before this since any temporary generated variables are not live out
    since python semantics determine that a range variable is assigned as it enters the body rather than crossing
    the latch, any live out enumerate/range arguments must be moved to the beginning of the loop body with new
    loop counters generated here.

    """
    # assume post ordering already handled
    counters = suitable_loop_counters(header, clobbers, liveouts)
    if counters:
        loop_index = counters.pop()
    else:
        loop_index = ""  # insert get unused name here
    move_to_body = []
    for target, value in header.walk_assignments():
        if target is loop_index:
            continue
        if isinstance(value, ir.Counter):
            if starts_from_zero(value):
                # fix positions later or make a pass that does it
                if has_unit_step(value):
                    stmt = ir.Assign(target, loop_index, header.pos)
                else:
                    stmt = ir.Assign(target, ir.BinOp(value.step, loop_index, "*"), header.pos)
            else:
                if has_unit_step(value):
                    stmt = ir.Assign(target, ir.BinOp(loop_index, value.start, "+"), header.pos)
                else:
                    stmt = ir.Assign(target, ir.BinOp(ir.BinOp(value.step, loop_index, "*"), value.start, "+"),
                                     header.pos)
            move_to_body.append(stmt)
        else:
            # track dim change
            # if this yields a scalar, then generate scalar assignment
            ref = array_refs.get(value)
            if ref is None:
                raise ValueError
            elif ref.ndims == 1:
                # if ndims(base) > 1, this should get a multidimensional subscript with respect to base ref
                move_to_body.append(ir.Assign(target, ir.Subscript(ref, loop_index), header.pos))
            else:
                # no explicit assignment, just check that this doesn't escape
                # need real error logging
                if target in liveouts:
                    raise ValueError
                if target in array_refs:
                    existing = array_refs[target]
                    assert (existing.ndims == ref.ndims - 1 and existing.dtype == ref.dtype)
                assert (target not in array_refs)
                array_refs[target] = ir.ArrayType(ref.ndims - 1, ref.dtype)
    repl_body = move_to_body.extend(header.body)
    return ir.ForLoop([(loop_index, ir.Counter(ir.IntNode(0), loop_index, ir.IntNode(1)),)], repl_body, header.pos)
    # Check each loop header for suitable loop counters, otherwise generate them
    # These are suitable to become the counter when the value cannot escape and it is not clobbered.


# Loop utilities

def post_order_loops(header):
    """
    Returns a sequence of loop headers for the nest enclosed by header, in post order.
    This could be a generator if we can guarantee a lack of modifications.
    """

    walkers = [(header, walk_branches(header))]
    ordered = []

    while walkers:
        try:
            stmt = next(walkers[-1][1])
            if stmt.is_loop_entry:
                walkers.append((stmt, walk_branches(stmt)))
                ordered.append(stmt)
        except StopIteration:
            loop, _ = walkers.pop()
            ordered.append(loop)

    return ordered


def is_divergent_loop(header):
    """
    Check whether the current loop is divergent.
    A divergent loop has a break statement at this level or a return statement at this level
    or within any nested loop.
    """
    # Post order loop headers
    # If any contain a return statement, that and enclosing are divergent
    # If the loop contains break, just that loop is divergent.

    for loop in post_order_loops(header):
        if loop is header:
            for stmt in loop.walk():
                if isinstance(stmt, (ir.Continue, ir.Break, ir.Return)):
                    return True
        else:
            for stmt in loop.walk():
                if isinstance(stmt, ir.Return):
                    return True
    return False


def is_innermost_loop(header):
    return not any(stmt.is_loop_entry for stmt in walk_branches(header))


def get_directly_nested_loops(header):
    """
    Return any loop headers ordered by appearance, which appear within this loop body,
    excluding further nesting levels.
    """

    return [stmt for stmt in walk_branches(header) if stmt.is_loop_entry]


def is_range_loop(header):
    """
    Determine if this is an IR representation of a range based for loop.
    This determines how escape analysis is applied.

    If the target value assigned by a range loop may escape the body of the loop where it's defined,
    it cannot be used as a loop counter, since the value will still increment on loop exit.

    """

    if isinstance(header, ir.ForLoop):
        if len(header.assigns) == 1:
            counter = header.assigns[0].value
            if isinstance(counter, ir.Counter):
                return counter.stop is None
    return False


def find_loopnest_conflicts(assign, entry):
    """
    Need to rethink this a bit...

    This is meant to help check how far a scalar load can be delayed in cases of packetization.

    Suppose we have something like the following.
    Before packetization, (a,b) are zipped with "a" being 1D and "b" being 2D, and "c" being read only
    over the inner loop or dead across the loop latch.

    Now also suppose after packetization, "c" varies by packet lane.
    Either we have to delay the load to "c" to the innermost loop after establishing loop bounds, or we have to
    expand "c" to an array form. In the latter case, "c" is privatized by the compiler later on. In the former,
    it's stuck with a static array.

    for i, (c, d) in enumerate(zip(a,b)):
        for j, e in enumerate(d):
            ...

    After lowering iterators, if everything is dead across the outer loop bound, we get something like this

    for i in range(min(len(a), len(b))):
        for j in range(b.shape[1]):
            c = a[i]
            e = b[i, j]
            ...

    The packetizer has to wrap any varying dataflow in addition to any varying control flow (not part of the example)

    since a,b are varying in this experiment,
    the packetizer wraps the innermost loop body

    for i in range(min(len(a), len(b))):
        for j in range(b.shape[1]):
            for k in range(p):
                c = a[i, j]
                e = b[i, j, k]
                ...

    If we can't move the load due to conflicts, we get something like this.
    (worse if the counting variables i,j escape their respective loops)

    for i in range(min(len(a), len(b))):
        c = a[i]
        for j in range(b.shape[1]):
            e = b[i, j]
            ...

    Then the packetizer has to generate an array for "c" rather than a scalar reference
    and refer to a subscript of "c" instead.

    This may perform worse.

    for i in range(min(len(a), len(b))):
        c = np.empty(p)
        for tmp in range(p):
            c[tmp] = a[i, j, tmp)]
        for j in range(b.shape[1]):
            for k in range(p):
               e = b[i, j]
               ...

    """
    pass


def get_named_iterables(header):
    """
    Get any named array references that are iterated over here.

    """

    iterables = set()

    for assign in header.assigns:
        if isinstance(assign.value, (ir.NameRef, ir.Subscript)):
            iterables.add(assign.value)

    return iterables


# Escape analysis

def find_live_across_latch(header):
    """
    Check whether anything may live across the loop latch.
    This determines whether a variable declaration could be set
    within the loop body.
    """
    # track read before write, assuming everything is bound (earlier check))

    check_possibly_live = ReachingCheck()
    maybe_live = check_possibly_live(header.body)
    if isinstance(header, ir.ForLoop):
        for assign in header.assigns:
            maybe_live.discard(assign.target)
    return maybe_live


# Assignment utilities

# Array handling utilities


def get_subscript_exprs(entry, enter_loops=False):
    """

    Walk everything reachable, starting from entry, marking anything that is a subscript
    expression.

    I may want to make a simpler walker.
    The key to this is to act like a visitor but yield nodes.


    Collect subscripted expressions.
    This is a precursor for bounds checking.

    Bounds checking proceeds by first determining whether all variables in an expression
    have an easily discernible evolution in the current scope. This generally means they are monotonic
    and we know the sign of the step value and only one assignment reaches any point in the loop body.

    If these conditions are met, we can first check whether these index expressions are already constrained
    by array bounds. If so, no instrumentation needed.

    Next, check if we can hoist determination of high and low bounds at runtime.
    This can be done with overflow checks.
    https://clang.llvm.org/docs/LanguageExtensions.html#checked-arithmetic-builtins

    If this fails, we fall back to brute force checks.
    This should log a performance warning in cases where it arises due to a failure to infer the evolution
    of patterned indexing.


    """

    pass


# General assignment utilities

def filter_by_target(assigns):
    """
    """

    subscripted = {}
    named = {}

    for assign in assigns:
        if assign.target.is_subscript:
            # by base should go here
            pass
        else:
            # by named should go here
            pass
