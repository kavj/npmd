import itertools
import operator

from typing import List

import npmd.ir as ir

from npmd.analysis import statements_match
from npmd.folding import make_logical_invert
from npmd.type_checks import TypeHelper
from npmd.symbol_table import SymbolTable
from npmd.traversal import get_statement_lists
from npmd.utils import is_entry_point


def unpack_branches(node: ir.IfElse) -> List[ir.IfElse]:
    """
    If this branch is perfectly nested, convert it to a case statement

    Note: this makes further if conversion much simpler
    Note: we may want to break along loop uniform conditions

    :param node:
    :return:
    """

    # Note, we only care about the first true instance of any unique predicate
    # Any duplicates become dead branches, and we need to avoid overwriting their corresponding entries

    next_stmt = node
    branches = []

    while True:
        branches.append(next_stmt)
        if len(next_stmt.else_branch) == 1 and isinstance(next_stmt.else_branch[0], ir.IfElse):
            next_stmt = next_stmt.else_branch[0]
        else:
            break
    return branches


def divergence_trivially_matches(a: ir.StmtBase, b: ir.StmtBase):
    if isinstance(a, ir.Break) and isinstance(b, ir.Break):
        return True
    elif isinstance(a, ir.Continue) and isinstance(b, ir.Continue):
        return True
    elif isinstance(a, ir.Return) and isinstance(b, ir.Return):
        return a.value == b.value


def inline_constant_branches(stmts: List[ir.StmtBase]):
    changed = False
    if any(isinstance(stmt, ir.IfElse) and isinstance(stmt.test, ir.CONSTANT) for stmt in stmts):
        repl = []
        for stmt in stmts:
            if isinstance(stmt, ir.IfElse) and isinstance(stmt.test, ir.CONSTANT):
                changed = True
                if operator.truth(stmt.test):
                    repl.extend(stmt.if_branch)
                else:
                    repl.extend(stmt.else_branch)
            else:
                repl.append(stmt)
        stmts.clear()
        stmts.extend(repl)
    return changed


def split_common_branch_divergence(stmts: List[ir.StmtBase]):
    """

    This moves control flow statements that appear at the beginning or end of both sides of a branch
    out of that branch.

    :param stmts:
    :return:
    """

    changed = False
    repl = []
    for stmt in stmts:
        if isinstance(stmt, ir.IfElse):
            if_branch = stmt.if_branch
            else_branch = stmt.else_branch

            while if_branch and else_branch:
                leading = if_branch[0]
                # If this can contain array valued expressions, then we don't want to impact statement fusion.
                # To handle the more general case in parallel loops, we have to also check that the branch value
                # isn't altered by code motion.
                if isinstance(leading, (ir.Break, ir.Continue))\
                        or (isinstance(leading, ir.Return) and isinstance(leading.value, (ir.CONSTANT, ir.NoneRef))):
                    if divergence_trivially_matches(leading, else_branch[0]):
                        repl.append(leading)
                        if_branch.pop(0)
                        else_branch.pop(0)
                        changed = True
                        continue
                break

            repl.append(stmt)

            while if_branch and else_branch:
                leading = else_branch[-1]
                if isinstance(leading, (ir.Break, ir.Continue))\
                        or (isinstance(leading, ir.Return) and isinstance(leading.value, (ir.CONSTANT, ir.NoneRef))):
                    if divergence_trivially_matches(leading, if_branch[-1]):
                        repl.append(leading)
                        if_branch.pop()
                        else_branch.pop()
                        changed = True
                        continue
                break
        else:
            repl.append(stmt)

    if changed:
        # if we hoisted anything from this branch, we have to transfer it.
        # this won't hoist nested branches or loops. It's mostly to extract
        # things like common break and continue statements, so that the CFG
        # doesn't have to deal the resulting divergence.
        stmts.clear()
        stmts.extend(repl)
    repl.clear()
    return changed


def denest_branches(stmts: List[ir.StmtBase]):
    """
    Given branches like

    if a:
       if b:
          ...
       else:
          noop
    else:
        ...

    transform to

    if a and b:
        ...
    else:
        ...

    """

    repl = []
    changed = False
    for stmt in stmts:
        if isinstance(stmt, ir.IfElse):
            if len(stmt.if_branch) == 1 \
                    and isinstance(stmt.if_branch[0], ir.IfElse) \
                    and len(stmt.if_branch[0].else_branch) == 0:
                changed = True
                nested = [stmt.test, stmt.if_branch[0].test]
                innermost_list = stmt.if_branch[0].if_branch
                while len(innermost_list) == 1 \
                        and isinstance(innermost_list[0], ir.IfElse) \
                        and len(stmt.if_branch[0].else_branch) == 0:
                    nested.append(innermost_list[0].test)
                    innermost_list = innermost_list[0].if_branch
                test = ir.AND(frozenset(nested))
                stmt.test = test
                stmt.if_branch = innermost_list
        else:
            repl.append(stmt)
    if changed:
        stmts.clear()
        stmts.extend(repl)
        repl.clear()
    return changed


def remove_dead_branches(stmts: List[ir.StmtBase]):
    """
    Removes branches in a perfectly nested branch sequence, that either have redundant conditions or
    conditions which are unreachable due to one of the prior conditions necessarily being true.

    if a:
       ...
    else:
       if b:
         ...
       else:
          if not a:
             ...
          else:
             # none of this is reachable because we have crossed both 'a' and 'not a'
             if c:
                ...
             else:
                ...


    if a:
       ...
    else:
       if b:
          ...
       else:
          not reachable, as we would have entered the first 'if a' if truth(a) == True
          if a:
              ...
          else:
              ...


    :param stmts:
    :return:
    """
    # Now look for branch nests where the predicate may be simplified due to repeated or inverse predicates
    changed = False
    for stmt in stmts:
        if isinstance(stmt, ir.IfElse):
            seen = set()
            branches = unpack_branches(stmt)
            for branch in branches:
                if isinstance(branch.test, ir.CONSTANT):
                    break
                elif branch.test in seen:
                    # If we have exclusive branches in a branch nest, and one of these branches
                    # already replicated this condition, then it must have already been evaluated false.
                    branch.test = ir.FALSE
                    changed = True
                    break
                else:
                    # we have already seen the inverse of this
                    inverted = make_logical_invert(stmt.test)
                    if inverted in seen:
                        # If the opposing condition was false on an enclosing branch, then this branch
                        # must be True
                        branch.test = ir.TRUE
                        changed = True
                        break
                    else:
                        seen.add(stmt.test)
    if changed:
        inline_constant_branches(stmts)
    return changed


def hoist_common_multi_branch(stmt: ir.IfElse, symbols: SymbolTable):
    """
    This needs to look for names which are all live in and common to a group of branches
    Probably need to annotate with liveness..

    :param stmt:
    :param symbols:
    :return:
    """
    pass


def hoist_common_single_branch(stmts: List[ir.StmtBase], symbols: SymbolTable):
    if not any(isinstance(stmt, ir.IfElse) for stmt in stmts):
        return False
    typer = TypeHelper(symbols)
    repl = []
    changed = False
    for stmt in stmts:
        if isinstance(stmt, ir.IfElse):
            if isinstance(stmt.test, ir.CONSTANT):
                # if this is a constant, we have a broken pipeline somewhere, as this needs to be handled via inlining
                raise TypeError('constant branches must already be handled via branch inlining before hoisting')
            repl_test = None
            while stmt.if_branch and stmt.else_branch and statements_match(stmt.if_branch[0], stmt.else_branch[0]):
                if is_entry_point(stmt.if_branch[0]):
                    break
                to_hoist = stmt.if_branch.pop(0)
                stmt.else_branch.pop(0)
                changed = True
                if repl_test is None:
                    if not (isinstance(to_hoist, (ir.Break, ir.Continue))
                            or (isinstance(to_hoist, ir.Return) and isinstance(to_hoist.value, ir.NoneRef))):
                        pos = stmt.pos
                        if isinstance(stmt.test, ir.NameRef):
                            test_name = symbols.make_versioned(stmt.test)
                        else:
                            test_type = typer(stmt.test)
                            test_name = symbols.make_unique_name_like('t', test_type)
                        repl_test = ir.Assign(test_name, stmt.test, pos)
                        repl.append(repl_test)
                repl.append(to_hoist)

            if repl_test is not None:
                stmt.test = repl_test

            while stmt.if_branch and stmt.else_branch and statements_match(stmt.if_branch[-1], stmt.else_branch[-1]):
                if is_entry_point(stmt.else_branch[-1]):
                    break
                # use the else branch for positional info on tail hoists
                repl.append(stmt.else_branch.pop())
                stmt.if_branch.pop()
                changed = True
        else:
            repl.append(stmt)
    if changed:
        stmts.clear()
        stmts.extend(repl)
        repl.clear()
    return changed


def refactor_branches(node: ir.Function, symbols: SymbolTable):
    """
    This repeatedly checks each statement list for refactorable branches, before iterating over nested
    components.
    :param node:
    :param symbols:
    :return:
    """
    for stmts in get_statement_lists(node):
        changed = True
        while changed:
            changed = inline_constant_branches(stmts)
            changed |= remove_dead_branches(stmts)
            changed |= split_common_branch_divergence(stmts)
            changed |= denest_branches(stmts)
            changed |= hoist_common_single_branch(stmts, symbols)
