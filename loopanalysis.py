import ir
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
            # something that isn't a normalized counter
            # replaces a normalized counter
            options.discard(target)
    return options


def post_ordering(header):
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
