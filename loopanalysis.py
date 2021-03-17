class ext_subscript:
    """
    stub for an extended subscript
    this is implementation only
    it's used to record all indices
    """
    pass


def collect_function_exits(func):
    # Find any places a function may need to exit.
    # This will eventually need to expand to include array allocation sites
    pass


def get_index_name(view):
    """
    Construct a name appropriate for iterating over this array dimension and stride pattern.
    This can be further versioned in cases of conflicts.

    It should be array name with _i, _j, _k, etc appended, starting from i.
    In cases of conflicts, these can be versioned, so we might have _i0, _j0, _k0.

    This is readable enough and doesn't greatly extend name lengths.

    """
    pass


def find_augmented(entry):
    """
    Find anything that is clobbered by name assignment or subscripted write, starting from entry.

    """

    pass


def find_unpacking_errors(header):
    # returns non-name reference targets, tuple to name assignments, attempted unpacking of name references
    pass


def get_header_targets(header):
    """
    Find anything assigned by the iterator field of a loop entry header, ignoring while headers.

    """
    pass


def get_named_header_iterables(header):
    """
    Get any named array references that are iterated over here.

    """
    pass


def is_divergent_loop(header):
    """
    Check whether the current loop is divergent.
    A divergent loop has a break statement at this level or a return statement at this level
    or within any nested loop.
    """

    pass


def contains_early_return(header):
    """
    Checks whether the loop body or any nested construct contains a return statement.
    This could be expanded later to check for possible exits due to array allocation.
    """

    pass


def find_varying_statements(entry):
    """
    Find statements that may vary in their output across a packet.

    """

    pass


def get_directly_nested_loops(header):
    """
    Return any loop headers ordered by appearance, which appear within this loop body,
    excluding further nesting levels.
    """
    pass


def post_order_loops(header):
    """
    Returns a sequence of loop headers for the nest enclosed by header, in post order.
    This could be a generator if we can guarantee a lack of modifications.
    """

    pass


def is_range_loop(header):
    """
    Determine if this is an IR representation of a range based for loop.
    This determines how escape analysis is applied.

    If the target value assigned by a range loop may escape the body of the loop where it's defined,
    it cannot be used as a loop counter, since the value will still increment on loop exit.

    """

    pass


def find_loopnest_conflicts(assign, entry):
    """
    Find statements, which conflict with the current assignment, reachable from
    the provided entry point.

    This is used to check for cases where a variable name is overwritten or it reads
    from an array that is modified at some point.

    This is used in conjunction with escape analysis to determine how far forward we can move an iterated
    assignment from an iterable, which may vary by packet lane.

    This is used to avoid widening scalars, which vary by packet lane and are retrieved in outer loops
    to short static arrays. In trivial cases, this allows us to privatize the scalars and leave the task of widening
    to the back end compiler.

    """
    pass


def find_loop_escapes(entry):
    """
    Find any variable reachable from entry that escapes the loop containing its first definition,
    which is visible from entry

    """
    pass


# write iterator transforms here and split later.
# This way we have analysis utility stubs in full view.


def may_have_negative_index(sub):
    """
    It's not yet clear what info will be needed here, even though we're only focusing on simple cases.

    Find cases where we cannot guarantee that the reaching value of an index is strictly positive.

    These are mostly cases where slicing contains subtraction or addition of possibly negative values.
    It may be worth warning. I haven't yet decided on exact default behavior. Wrapping doesn't mix with simd.

    """
    pass


def collect_iterator_strides(it):
    """
    Map iterables to strides.
    This helps with reduction to simpler range loops.
    It's worth tracking cases of dead variables wrapped in enumerate to constrain range, as we can preserve
    the constraint without factoring this into the eventual range stride.

    """
    # Checks whether 1 or more iterables in an iterator nest uses a non-unit stride
    pass


def collect_iterator_offsets(it):
    """
    Map iterables to their starting offsets, with respect to their current view.
    This is used to determine if offsets can be folded into range.start.

    """

    pass


def get_iterable_length(it):
    """
    This is used to define a canonical iterable length expression for an iterable nest.
    Since it's deterministic, we can check whether such an expression is already bound to some variable
    name, using its hash.

    """
    pass


def range_convert_loop(header, escapes):
    """
    Convert from an arbitrary iterator representation to a range loop, moving any assignments to
    loop body subscripts.

    """

    pass


def find_implem_name_conflicts(ctx):
    """
    This doesn't belong here, but it checks for naming conflicts on array parameters.

    Internally I'm settling on conventions for things like index naming, based on what they index.


    """

    pass


def flatten_index_expr(sub):
    """
    Turn a subscripted expression, which may dereference a strided or sliced array into an expression depicting
    a flat index offset.

    """
    pass
