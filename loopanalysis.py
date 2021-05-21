import ir
from visitor import walk_branches

"""
All utilities used
"""


# function utilities

def get_loop_constraints(node: ir.ForLoop):
    array_constraints = set()
    scalar_constraints = set()
    for _, iterable in node.walk_assignments():
        if isinstance(iterable, ir.Counter):
            if iterable.stop is not None:
                #  keep step since we may not know sign
                scalar_constraints.add((iterable.stop, iterable.step))
        else:
            array_constraints.add(iterable)
    return array_constraints, scalar_constraints


# info was typing.Dict[ir.NameRef, VarInfo]
def find_valid_loop_counter(header: ir.ForLoop, info, no_escape):
    for target, iterable in header.walk_assignments():
        if target not in no_escape:
            continue
        if isinstance(iterable, ir.Counter):
            vi = info[target]
            if (iterable.start == ir.IntNode(0)
                    and iterable.step == ir.IntNode(1)
                    and vi.min_dim_constraint == 0
                    and not vi.written
                    and not vi.augmented):
                return target


def find_nested_loops(stmts):
    nested = []
    for stmt in stmts:
        if isinstance(stmt, ir.IfElse):
            nested.extend(find_nested_loops(stmt.if_branch))
            nested.extend(find_nested_loops(stmt.else_branch))
        elif isinstance(stmt, (ir.ForLoop, ir.WhileLoop)):
            nested.append(stmt)
    return nested


def generate_header_assignments(header: ir.ForLoop, loop_index):
    """
    Generate a sequence of assignments that should move to the loop body.
    If any target is to be reused as a loop index, we skip its explicit loop body assignment,
    eg. don't generate an assignment of the form "i = i"

    """
    assigns = []

    # ignore pos for now, this would need to update line number and generate new positional info
    # it really needs a visitor to do this properly though.
    current_pos = header.pos

    for target, iterable in header.walk_assignments():
        if isinstance(iterable, ir.Counter):
            # for simplicity, make new loop index by default
            # This requires far fewer checks..
            expr = loop_index
            if iterable.step != ir.IntNode(1):
                expr = ir.BinOp(iterable.step, expr, "*")
            if iterable.start != ir.IntNode(0):
                expr = ir.BinOp(expr, iterable.start, "+")
            assigns.append(ir.Assign(target, expr, current_pos))
        else:
            assigns.append(ir.Assign(target, ir.Subscript(iterable, loop_index), current_pos))

    return assigns


def generate_loop_assignments(header, loop_index):
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
