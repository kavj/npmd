# Todo: how would reductions work?
from collections import defaultdict
from typing import Dict, Generator, List, Tuple

import lib.ir as ir

from lib.analysis import find_assigned_or_augmented, get_target_name
from lib.traversal import get_statement_lists
from lib.utils import unpack_iterated


"""
The first issue is, how do we define an initial ergonomic vectorization pattern?



"""


def get_write_statements_per_variable(node: ir.ForLoop) -> Generator[Tuple[ir.NameRef, ir.NameRef], None, None]:
    for target, _ in unpack_iterated(node.target, node.iterable):
        yield get_target_name(node.target), node
    for stmts in get_statement_lists(node, enter_nested_loops=False):
        for stmt in stmts:
            if isinstance(stmt, (ir.Assign, ir.InPlaceOp)):
                yield get_target_name(stmt.target), stmt


def get_induction_parameters(node: ir.ForLoop):
    """
    This grabs all variables which can safely be treated as induction for a loop.
    At this time, it'll only catch variables bound once
    :return:
    """

    by_target = defaultdict(list)
    for target, stmt in get_write_statements_per_variable(node):
        by_target[target].append(stmt)

    induction = {}
    for target, iterable in unpack_iterated(node.target, node.iterable):
        if isinstance(iterable, ir.AffineSeq) and len(by_target[target]) == 1:
            induction[target] = iterable
    # get simple induction first.
    # TODO: Make a second pass for those that can be declared induction by forward substitution
    #  This has the caveat that we need
    for stmts in get_statement_lists(node):
        for stmt in stmts:
            if isinstance(stmt, (ir.Assign, ir.InPlaceOp)) and isinstance(stmt.target, ir.NameRef):
                target = stmt.target
                if len(by_target[target]) == 1:
                    # TODO: handle subtract as well?
                    if isinstance(stmt.value, ir.ADD):
                        other_params = [subexpr for subexpr in stmt.value.subexprs if subexpr != target]
                        if not other_params:
                            # a + a is not induction
                            continue


def gather_access_patterns(node: ir.ForLoop):
    """
    Get any step patterns used
    :return:
    """

    # we need to grab read patterns based on each variable
    # basically get index in each case unless a for loop with name target
    # in that case use standard 1::

    refs = defaultdict(list)
    induction = set()
    pass


def get_trip_count():
    """
    get strongest definition for trip count available
    :return:
    """
    pass


def sequence_nested_loops():
    """
    Get nested loops in depth first order with nesting level.
    This way we can easily determine nesting structure.
    :return:
    """
    pass


def find_allocations():
    """
    No allocations inside vectorization target regions
    :return:
    """

    pass


def is_simple_if_statement():
    """
    Returns True if this is a branch point with a single block in each side
    :return:
    """

    pass


def is_flattenable_branch_nest():
    """
    We can flatten if this is a perfect nest of if elifs
    :return:
    """
    pass


def get_indexing():
    """
    This should pair variables to expressions used to read and write them.
    We impose enough restrictions here to keep things sensible. Since we want to fuse various
    vectorized loops, it makes sense to built vectorization infrastructure around the loops themselves
    and transformed iterators.
    :return:
    """
    pass


# Todo: need some dependence checking


def find_dependence_conflicts():
    """
    Find things writing out of turn
    :return:
    """
    pass


def map_targets():
    """
    Get escaping definitions and corresponding predicates for all live out targets
    :return:
    """
    pass


def is_min_max_coercible():
    """
    checks whether this can be reduced to some variation of a if a >< b else b
    :return:
    """
    pass
