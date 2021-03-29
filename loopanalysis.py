import typing

import ir
from reachingcheck import ReachingCheck
from visitor import walk_branches

"""
All utilities used
"""


# function utilities

def collect_function_exits(func):
    """
    Find places the function may exit.
    This checks for exit by dropping out of scope, explicit return statements, dynamic buffer allocation,
    and bounds checking.

    """
    pass


def find_varying_statements(entry):
    """
    Find statements that may vary in their output across a packet.

    """

    pass


# Loop utilities


def find_simple_inductions(header):
    pass


def get_counters(header):
    """
    Get any counting iterables and their targets, defined here.
    When simplifying loops to range form, this helps determine if any existing
    counter may be reused.

    """
    pass


def build_scevs(assign):
    """
    Build a scalar evolution for the case where a single assignment
    to an index variable is used within the loop of interest.

    """

    # post order loop nest
    # For each loop from inner to outer
    # find counters
    # find additional clobbers
    # mark step and depth.
    # patch outer loop evols into inner

    pass


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


def collect_iterator_offsets(it):
    """
    Map iterables to their starting offsets, with respect to their current view.
    This is used to determine if offsets can be folded into range.start.

    """

    pass


def get_iterator_stride(it):
    """
    unit step, non-unit constant step, symbolic step,

    """
    pass


def get_iterable_length(it):
    """
    This is used to define a canonical iterable length expression for an iterable nest.
    Since it's deterministic, we can check whether such an expression is already bound to some variable
    name, using its hash.

    """
    pass


def find_unpacking_mismatches(node):
    """
    Old version is invalid now due to more aggressive serialization.
    """

    errors = []

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


def flatten_index_expr(sub):
    """
    Turn a subscripted expression, which may dereference a strided or sliced array into an expression depicting
    a flat index offset.

    """
    pass


def may_be_negative(sub):
    """
    It's not yet clear what info will be needed here, even though we're only focusing on simple cases.

    Find cases where we cannot guarantee that the reaching value of an index is strictly positive.

    These are mostly cases where slicing contains subtraction or addition of possibly negative values.
    It may be worth warning. I haven't yet decided on exact default behavior. Wrapping doesn't mix with simd.

    """
    pass


def collect_subscripts():
    """
    Gathered to try to prove away the need for bounds checking within a loop.

    """
    pass


def find_augmenting(assigns):
    """
    Find anything that is clobbered by name assignment or subscripted assignment, starting from entry.

    """

    pass


def get_array_index_name(view):
    """
    Construct an abi specific unversioned name for this array dimension and stride pattern.

    It should be array name with _i, _j, _k, etc appended, starting from i.
    In cases of conflicts, these can be versioned, so we might have _i0, _j0, _k0.

    This is readable enough and doesn't greatly extend name lengths.

    """
    pass


class BoundsCheck:
    """
    A stub class for an instrumentation node, indicating a bounds check barrier.
    This ia an IR node to hide its implementation and lowering from other analyses.
    It isn't an external call, because it may divert control flow and I don't want to deal with noreturn functions.

    This should be added prior to lowering iterators, so as not to add unnecessary instrumentation.

    """

    upper_constr: typing.Union[ir.NameRef, ir.IntNode]
    lower_constr: typing.Union[ir.NameRef, ir.IntNode]
    failure_msg: str


def insert_bounds_check():
    pass


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


# find single assignments should proceed using results
# of filter by target


def find_implem_name_conflicts(ctx):
    """
    There will be abi specific details for index naming and some other things.
    We'll need to check for cases where these conflict with names found in the original symbol
    table.

    """

    pass
