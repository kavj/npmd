import math
import operator

import ir
from visitor import walk_branches


"""
All utilities used
"""


# function utilities

def is_innermost(header):
    return not any(stmt.is_loop_entry for stmt in walk_branches(header))


def unwrap_loop_body(node):
    return node.body if isinstance(node, (ir.ForLoop, ir.WhileLoop)) else node


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
    for stmt in unwrap_loop_body(entry):
        if isinstance(stmt, ir.Assign):
            if isinstance(stmt.target, ir.NameRef):
                written_vars.add(stmt.target)
            else:
                written_exprs.add(stmt.target)
    return written_vars, written_exprs


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


def get_simple_range_step_count(counter):
    """
    return range step count if it requires little to no computation
    If step count is difficult to determine, return None.

    This has a very large number of possibilities, some of which can be expensive to compute at runtime
    if lowered to compute an exact count rather than break on induction bounds

    """
    if isinstance(counter.step, ir.IntNode):
        if counter.step.value == 1:
            if isinstance(counter.stop, ir.IntNode) and isinstance(counter.start, ir.IntNode):
                diff = ir.IntNode(operator.sub(counter.stop, counter.start)
                                  if counter.stop.value > counter.start.value else 0)
            else:
                diff = (ir.IfExpr(ir.BinOp(counter.stop, counter.start, ">"),
                                  ir.BinOp(counter.stop, counter.start, "-"), ir.IntNode(0)))
            return diff
        elif counter.step.value == 0:
            raise ValueError("range step of size 0")
        elif isinstance(counter.start, ir.IntNode) and isinstance(counter.stop, ir.IntNode):
            if (counter.stop.value > counter.start.value) and (counter.step.value > 0):
                diff = operator.sub(counter.stop.value, counter.start.value)
                count = math.ceil(operator.div(diff, counter.step.value))
            elif (counter.stop.value < counter.start.value) and (counter.step.value < 0):
                diff = operator.sub(counter.start.value, counter.stop.value)
                count = math.ceil(operator.div(diff, operator.abs(counter.step.value)))
            else:
                count = 0
            return count
        else:
            # may require runtime division, lower to affine inducation instead
            return None


def contained_loops(stmts):
    nested = []
    for stmt in walk_branches(stmts):
        if isinstance(stmt, (ir.ForLoop, ir.WhileLoop)):
            nested.append(stmt)
    return nested


def target_iterable_conflicts(header):
    """
    Python permits a target to clobber an iterable name, since the value obtained for an iterable
    is based on whatever the name or expression evaluates to at the time the loop is first encountered.
    This violates static typing though, thus it must be checked.
    """

    targets = set()
    iterables = set()
    duplicate_targets = set()
    for target, iterable in header.walk_assignments():
        if target in targets:
            duplicate_targets.add(target)
        targets.add(target)
        iterables.add(iterable)
    conflicts = targets.intersection(iterables)
    return conflicts, duplicate_targets


def final_header_assignments(header):
    assigns = {}
    for target, iterable in header.walk_assignments():
        assigns[target] = iterable
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
            # targets which accept more than one assignment are excluded.
            options.discard(target)
    return options


def post_ordering(header):
    """
    returns post ordering of a loop nest
    consecutive loops at the same nesting level appear as nested sequences
    """
    nested = contained_loops(header.body)
    nested_count = len(nested)
    if nested_count == 0:
        nested = [header]
    elif nested_count == 1:
        nested.append(header)
    else:
        nested = [nested, header]
    return nested


def loop_body_exits(body):
    exits = []
    for stmt in walk_branches(body):
        if isinstance(stmt, (ir.Break, ir.Return)):
            exits.append(stmt)
    return exits
