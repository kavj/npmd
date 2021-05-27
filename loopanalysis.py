import ir
from reachingcheck import ReachingCheck
from visitor import walk_branches


"""
All utilities used
"""


# function utilities

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


def find_nested(stmts):
    nested = []
    for stmt in stmts:
        if isinstance(stmt, ir.IfElse):
            nested.extend(find_nested(stmt.if_branch))
            nested.extend(find_nested(stmt.else_branch))
        elif isinstance(stmt, (ir.ForLoop, ir.WhileLoop)):
            nested.append(stmt)
    return nested


def generate_header_assignments(header, loop_index):
    assigns = []
    for target, iterable in header.walk_assignments():
        if target is loop_index:
            continue
        if isinstance(iterable, ir.Counter):
            affine = loop_index
            if iterable.step != ir.IntNode(1):
                affine = ir.BinOp(iterable.step, affine, "*")
            if iterable.start != ir.IntNode(0):
                affine = ir.BinOp(affine, iterable.start, "+")
            assigns.append(ir.Assign(target, affine, header.pos))
        else:
            # must be array type
            ndims = iterable.ndims
            if ndims < 0:
                # Todo: fix up compile time error checks
                raise ValueError("oversubscripted array")
            # Todo: fix up position on rewrites
            assigns.append(ir.Assign(target, ir.ViewRef(iterable, loop_index), header.pos))
    return assigns


# Any of these that don't escape the loop body and aren't clobbered inside it
# can be set as loop index.
def normalized_counters(header):
    options = set()
    for target, iterable in header.walk_assignments():
        if (isinstance(iterable, ir.Counter)
                and iterable.start == ir.IntNode(0)
                and iterable.step == ir.IntNode(1)):
            options.add(target)
        elif target in options:
            # targets with multiple header assignments are not
            # considered normalized
            options.discard(target)
    return options


def post_ordering(header):
    """
    returns post ordering of a loop nest
    consecutive loops at the same nesting level appear as nested sequences
    """
    nested = find_nested(header.body)
    nested_count = len(nested)
    if nested_count == 0:
        nested = [header]
    elif nested_count == 1:
        nested.append(header)
    else:
        nested = [nested, header]
    return nested


def exits_from_body(body):
    """
    Checks whether break or return is found in this statement list, entering ifelse branches
    but not nested loops.
    """
    for stmt in body:
        if isinstance(stmt, (ir.Break, ir.Return)):
            return True
        elif isinstance(stmt, ir.IfElse):
            if exits_from_body(stmt.if_branch) or exits_from_body(stmt.else_branch):
                return True
    return False


def is_innermost(header):
    return not any(stmt.is_loop_entry for stmt in walk_branches(header))


def contained_writes(entry):
    """
    return (writes_normal, writes_subscript)

    This way we can determine if it's legal to optimize to identity

    This is useful for a lot of things. For example, we might have a uniform assignment
    in a varying branch, which as a result still requires some kind of predication

    """
    # check separately writes on true branch, writes on false branch
    written_vars = set()
    written_exprs = set()
    for stmt in entry:
        if isinstance(stmt, ir.Assign):
            if isinstance(stmt.target, ir.NameRef):
                written_vars.add(stmt.target)
            else:
                written_exprs.add(stmt.target)
    return written_vars, written_exprs


def find_nested_loop_escapes(header):
    """
    This is a rough check as to whether anything could leak. Reaching check will only not declare anything
    bound in an outer loop due to an assignment in an inner one, so it's safe to use here.
    If something isn't declared prior to the loop and isn't marked as escaping, we can move the declaration
    inside, which also means potentially reusing enumerate variables as loop

    This isn't completely exact. For example, multiple inner loops at a given nesting level may induce false positives.

    """
    rc = ReachingCheck()
    post_ordered = post_ordering(header)
    writes, _ = contained_writes(post_ordered[0])
    escapees = []
    writes = []
    w, _ = contained_writes(post_ordered[0])
    writes.append(w)
    cumulative_writes = w.copy()
    for loop_header in post_ordered[1:]:
        # find uses that may precede assignment
        maybe_live_on_entry = rc(loop_header)
        # this captures anything that may be read before written in this scope but not globally
        # global checks for unboundedness take place early, thus this is probably good enough
        escapees.append(maybe_live_on_entry.intersection(cumulative_writes))
        w, _ = contained_writes(loop_header)
        cumulative_writes.update(w)
    return post_ordered, escapees, writes
