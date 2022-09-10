import itertools
import math
import numpy

from collections import defaultdict
from functools import singledispatchmethod
from typing import Iterable, List, Set, Tuple, Union

import npmd.ir as ir

from npmd.analysis import compute_element_count, get_branch_predicate_pairs
from npmd.blocks import build_function_graph, get_loop_exit_block
from npmd.errors import CompilerError
from npmd.folding import fold_constants, simplify_arithmetic
from npmd.liveness import find_live_in_out, remove_dead_branches, remove_unreachable_statements
from npmd.lvn import rewrite_expr
from npmd.pretty_printing import PrettyFormatter
from npmd.symbol_table import SymbolTable
from npmd.traversal import get_statement_lists, walk_nodes
from npmd.type_checks import is_integer, TypeHelper
from npmd.utils import extract_name, unpack_iterated


def invalid_loop_iterables(node: ir.ForLoop, symbols: SymbolTable):
    type_checker = TypeHelper(symbols)
    for _, iterable in unpack_iterated(node.target, node.iterable):
        if isinstance(iterable, ir.AffineSeq):
            for subexpr in iterable.subexprs:
                # Todo: raise here instead?
                if not is_integer(type_checker(subexpr)):
                    yield subexpr
        elif isinstance(iterable, ir.Subscript):
            if isinstance(iterable.index, ir.Slice):
                for subexpr in iterable.index.subexprs:
                    if not is_integer(type_checker(subexpr)):
                        yield subexpr
            elif not is_integer(type_checker(iterable.index)):
                yield iterable.index
        else:
            t = type_checker(iterable)
            if not isinstance(t, ir.ArrayType):
                yield iterable


def unpack_min_terms(terms: Iterable[ir.ValueRef], type_check: TypeHelper) -> List[ir.ValueRef]:
    repl = []
    queued = [*terms]
    seen = set()
    while queued:
        term = queued.pop()
        if term in seen:
            continue
        seen.add(term)
        if isinstance(term, (ir.MIN, ir.MinReduction)):
            # safety check needed so we don't reorder
            if all(is_integer(type_check(subexpr)) for subexpr in term.subexprs):
                queued.extend(term.subexprs)
            else:
                repl.append(term)
        else:
            repl.append(term)
    return repl


def unpack_max_terms(terms: Iterable[ir.ValueRef], type_check: TypeHelper) -> List[ir.ValueRef]:
    repl = []
    queued = [*terms]
    seen = set()
    while queued:
        term = queued.pop()
        if term in seen:
            continue
        seen.add(term)
        if isinstance(term, (ir.MAX, ir.MaxReduction)):
            if all(is_integer(type_check(subexpr)) for subexpr in term.subexprs):
                queued.extend(term.subexprs)
            else:
                repl.append(term)
        else:
            repl.append(term)
    return repl


def wrap_max_reduction(terms: Iterable[ir.ValueRef], type_check: TypeHelper) -> ir.ValueRef:
    ordered = frozenset(unpack_max_terms(terms, type_check))
    if len(ordered) == 1:
        return next(iter(ordered))
    elif len(ordered) == 2:
        return ir.MAX(*ordered)
    else:
        return ir.MaxReduction(ordered)


def wrap_min_reduction(terms: List[ir.ValueRef], type_check: TypeHelper) -> ir.ValueRef:
    ordered = frozenset(unpack_min_terms(terms, type_check))
    if len(ordered) == 1:
        return next(iter(ordered))
    elif len(ordered) == 2:
        return ir.MIN(*ordered)
    else:
        return ir.MinReduction(*ordered)


class NormalizeMinMax:
    """
    Reorders in integer cases, where ordering is arbitrary. Also consolidates identical terms in such cases.
    """

    def __init__(self, symbols: SymbolTable):
        self.symbols = symbols
        self.typer = TypeHelper(symbols)

    def __call__(self, node: ir.ValueRef) -> ir.ValueRef:
        return self.visit(node)

    @singledispatchmethod
    def visit(self, node):
        msg = f'No method to apply normalize integral to "{node}".'
        raise TypeError(msg)

    @visit.register
    def _(self, node: ir.ValueRef):
        return node

    @visit.register
    def _(self, node: ir.Expression):
        repl = node.reconstruct(*(self.visit(subexpr) for subexpr in node.subexprs))
        return repl

    @visit.register
    def _(self, node: ir.MAX):
        repl = ir.MAX(*(self.visit(subexpr) for subexpr in node.subexprs))
        if any(not is_integer(self.typer(subexpr)) for subexpr in repl.subexprs):
            return repl
        return wrap_max_reduction(repl.subexprs, self.typer)

    @visit.register
    def _(self, node: ir.MIN):
        repl = ir.MIN(*(self.visit(subexpr) for subexpr in node.subexprs))
        if any(not is_integer(self.typer(subexpr)) for subexpr in repl.subexprs):
            return repl
        return wrap_min_reduction(repl.subexprs, self.typer)

    @visit.register
    def _(self, node: ir.MinReduction):
        # we don't pay attention to ordering on these
        repl = ir.MinReduction(*(self.visit(subexpr) for subexpr in node.subexprs))
        return wrap_min_reduction(repl.subexprs, self.typer)

    @visit.register
    def _(self, node: ir.MaxReduction):
        repl = ir.MaxReduction(*(self.visit(subexpr) for subexpr in node.subexprs))
        return wrap_max_reduction(repl.subexprs, self.typer)


def serialize_min_max(node: Union[ir.MinReduction, ir.MaxReduction]):
    """
    Min max serialize without scraping all pairs
    :param node:
    :return:
    """
    if isinstance(node, ir.MinReduction):
        reducer = ir.MIN
    elif isinstance(node, ir.MaxReduction):
        reducer = ir.MAX
    else:
        msg = f"serializer requires min or max reduction. Received {node}."
        raise TypeError(msg)

    terms = list(node.subexprs)

    # serialize any nested terms
    for index, term in enumerate(terms):
        if isinstance(term, (ir.MaxReduction, ir.MinReduction)):
            terms[index] = serialize_min_max(term)

    num_terms = len(terms)
    if num_terms == 1:
        value = terms[0]
        return value
    if num_terms % 2:
        tail = terms[-1]
        terms = terms[:-1]
    else:
        tail = None
    step_count = math.floor(math.log2(len(terms)))

    for i in range(step_count):
        terms = [reducer(left, right) for left, right
                 in zip(itertools.islice(terms, 0, None, 2), itertools.islice(terms, 1, None, 2))]
    assert len(terms) == 1
    reduced = terms[0]
    if tail is not None:
        reduced = reducer(reduced, tail)
    return reduced


def stmt_matches(a: ir.StmtBase, b: ir.StmtBase):
    if isinstance(a, ir.Break) and isinstance(b, ir.Break):
        return True
    elif isinstance(a, ir.Continue) and isinstance(b, ir.Continue):
        return True
    elif isinstance(a, ir.Return) and isinstance(b, ir.Return):
        return a.value == b.value


def hoist_control_flow(node: ir.Function):
    """

    This moves control flow statements that appear at the beginning or end of both sides of a branch
    out of that branch.

    :param node:
    :return:
    """

    for stmt_list in get_statement_lists(node):
        if any(isinstance(stmt, ir.IfElse) for stmt in stmt_list):
            hoisted = False
            repl = []
            for stmt in stmt_list:
                if isinstance(stmt, ir.IfElse):
                    if_branch = stmt.if_branch
                    else_branch = stmt.else_branch

                    while if_branch and else_branch:
                        leading = if_branch[0]
                        if isinstance(leading, (ir.Break, ir.Continue, ir.Return)):
                            if stmt_matches(leading, else_branch[0]):
                                repl.append(leading)
                                if_branch.pop(0)
                                else_branch.pop(0)
                                hoisted = True
                                continue
                        break

                    repl.append(stmt)

                    while if_branch and else_branch:
                        leading = else_branch[-1]
                        if isinstance(leading, (ir.Break, ir.Continue, ir.Return)):
                            if stmt_matches(leading, if_branch[-1]):
                                repl.append(leading)
                                if_branch.pop()
                                else_branch.pop()
                                hoisted = True
                                continue
                        break
                else:
                    repl.append(stmt)

            if hoisted:
                # if we hoisted anything from this branch, we have to transfer it.
                # this won't hoist nested branches or loops. It's mostly to extract
                # things like common break and continue statements, so that the CFG
                # doesn't have to deal the resulting divergence.
                stmt_list.clear()
                stmt_list.extend(repl)
            repl.clear()


def split_common_branch_statements(node: ir.Function):
    """
    splits out statements that are part of a prefix or suffix common to all branches of the branch statement.
    :param node:
    :return:
    """
    for stmt_list in get_statement_lists(node):
        hoisted = False
        repl = []
        if any(isinstance(stmt, ir.IfElse) for stmt in stmt_list):
            for stmt in stmt_list:
                if isinstance(stmt, ir.IfElse):
                    # first look for a shared prefix
                    if_branch = stmt.if_branch
                    else_branch = stmt.else_branch

                    while if_branch and else_branch:
                        if stmt_matches(if_branch[0], else_branch[0]):
                            hoisted = True
                            common = if_branch.pop(0)
                            repl.append(common)
                            else_branch.pop(0)
                        else:
                            break

                    # now append the rest of the branch
                    repl.append(stmt)

                    # now look for a remaining common suffix
                    while if_branch and else_branch:
                        if stmt_matches(if_branch[-1], else_branch[-1]):
                            hoisted = True
                            common = else_branch.pop()
                            repl.append(common)
                            if_branch.pop()
                        else:
                            break

        if hoisted:
            # if we hoisted anything from this branch, we have to transfer it.
            # this won't hoist nested branches or loops. It's mostly to extract
            # things like common break and continue statements, so that the CFG
            # doesn't have to deal the resulting divergence.
            stmt_list.clear()
            stmt_list.extend(repl)
            repl.clear()


def add_trivial_return(node: ir.Function):
    """
    Method to add a trivial return to function.
    This will typically get cleaned up later if it's not reachable.
    :param node:
    :return:
    """

    if len(node.body) > 0:
        if isinstance(node.body[-1], ir.Return):
            return
        last = node.body[-1]
        pos = ir.Position(last.pos.line_end + 1, last.pos.line_end + 1, 1, 40)
        node.body.append(ir.Return(ir.NoneRef(), pos))
    else:
        pos = ir.Position(-1, -1, 1, 40)
        node.body.append(ir.Return(ir.NoneRef(), pos))


def replace_call(node: ir.Call):
    func_name = extract_name(node)
    if func_name == "numpy.ones":
        node = ir.ArrayInitializer(*node.args, ir.One)
    elif func_name == "numpy.zeros":
        node = ir.ArrayInitializer(*node.args, ir.Zero)
    elif func_name == "numpy.empty":
        node = ir.ArrayInitializer(*node.args, ir.NoneRef())
    elif func_name == "zip":
        node = ir.Zip(*node.args)
    elif func_name == "enumerate":
        node = ir.Enumerate(*node.args)
    elif func_name == "range":
        nargs = len(node.args)
        if nargs == 3:
            node = ir.AffineSeq(*node.args)
        elif nargs == 2:
            start, stop = node.args
            node = ir.AffineSeq(start, stop, ir.One)
        elif nargs == 1:
            stop, = node.args
            node = ir.AffineSeq(ir.Zero, stop, ir.One)
        else:
            pf = PrettyFormatter()
            msg = f"bad arg count for call to range {pf(node)}"
            raise CompilerError(msg)
    elif func_name == "len":
        assert len(node.args) == 1
        node = ir.SingleDimRef(node.args[0], ir.Zero)
    elif func_name == 'min':
        assert len(node.args) > 1
        terms = ir.MIN(node.args[0], node.args[1])
        for arg in itertools.islice(node.args, 2, None):
            terms = ir.MIN(terms, arg)
        node = terms
    elif func_name == 'max':
        assert len(node.args) > 1
        terms = ir.MAX(node.args[0], node.args[1])
        for arg in itertools.islice(node.args, 2, None):
            terms = ir.MAX(terms, arg)
        node = terms
    return node


def preheader_rename_parameters(node: ir.ForLoop):
    """

    :param node:
    :return:
    """

    # Todo: we'll need a check for array augmentation as well later to see whether it's easily provable
    # that writes match the read pattern at entry

    clobbered_in_body = set()
    augmented_in_body = set()
    for stmt in walk_nodes(node.body):
        if isinstance(stmt, ir.Assign):
            if isinstance(stmt.target, ir.NameRef):
                clobbered_in_body.add(stmt.target)
            else:
                augmented_in_body.add(stmt.target)

        elif isinstance(stmt, ir.InPlaceOp):
            if isinstance(stmt.target, ir.NameRef):
                augmented_in_body.add(stmt.target)
            else:
                assert isinstance(stmt.target, ir.Subscript)
                augmented_in_body.add(stmt.target.value)

        elif isinstance(stmt, ir.ForLoop):
            for target, _ in unpack_iterated(stmt.target, stmt.iterable):
                assert isinstance(target, ir.NameRef)
                clobbered_in_body.add(target)

    must_rename = set()

    for target, iterable in unpack_iterated(node.target, node.iterable):
        if isinstance(iterable, ir.AffineSeq):
            for subexpr in iterable.subexprs:
                if isinstance(subexpr, ir.Expression):
                    # If this is an expression, it could have array access and that could
                    # change. For simplicity, just match to a variable in preheader
                    must_rename.add(subexpr)
                elif isinstance(subexpr, ir.NameRef):
                    if subexpr in clobbered_in_body or subexpr in augmented_in_body:
                        must_rename.add(subexpr)
        elif isinstance(iterable, ir.Expression):
            must_rename.add(iterable)
        else:
            assert isinstance(iterable, ir.NameRef)
            if iterable in clobbered_in_body:
                # ignore augmented only
                must_rename.add(iterable)
    return must_rename


def denest_branches(func: ir.Function):
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
    for stmt_list in get_statement_lists(func):
        changed = False
        for stmt in stmt_list:
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
            stmt_list.clear()
            stmt_list.extend(repl)
            repl.clear()


def leading_statements_match(stmt_lists: List[List[ir.StmtBase]]):
    if len(stmt_lists) == 0:
        return False
    elif len(stmt_lists[0]) == 0:
        return False
    lead = stmt_lists[0][0]
    for stmt_list in stmt_lists:
        if len(stmt_list) == 0 or not stmt_matches(lead, stmt_list[0]):
            return False
    return True


def trailing_statements_match(stmt_lists: List[List[ir.StmtBase]]):
    if len(stmt_lists) == 0:
        return False
    elif len(stmt_lists[-1]) == 0:
        return False
    lead = stmt_lists[-1][-1]
    for stmt_list in stmt_lists:
        if len(stmt_list) == 0 or not stmt_matches(lead, stmt_list[-1]):
            return False
    return True


def hoist_from_branch_nest(func: ir.Function):
    for stmt_list in get_statement_lists(func):
        repl = []
        maybe_hoisted = False
        for stmt in stmt_list:
            if isinstance(stmt, ir.IfElse):
                maybe_hoisted = True
                branches_only = [branch for (predicate, branch, pos) in get_branch_predicate_pairs(stmt)]
                while leading_statements_match(branches_only):
                    repl.append(branches_only[0][0])
                    for b in branches_only:
                        b.pop(0)
                repl.append(stmt)
                while trailing_statements_match(branches_only):
                    repl.append(branches_only[-1][-1])
                    for b in branches_only:
                        b.pop()
            else:
                repl.append(stmt)
        if maybe_hoisted:
            stmt_list.clear()
            stmt_list.extend(repl)


def normalize_flow(func: ir.Function, symbols: SymbolTable):
    add_trivial_return(func)
    remove_unreachable_statements(func, symbols)
    remove_unreachable_statements(func, symbols)
    denest_branches(func)
    remove_dead_branches(func)
    denest_branches(func)
    hoist_from_branch_nest(func)


def rename_clobbered_loop_parameters(func: ir.Function, symbols: SymbolTable, verbose: bool = True):
    """
    This makes a copy of cases where the name used by the loop header is overwritten in the body,
    which makes analyzing their liveness much less confusing, particularly in the absence of LCSSA.

    :param func:
    :param symbols:
    :param verbose:
    :return:
    """
    # Todo: do we actually want this as a CFG dependent pass?
    typer = TypeHelper(symbols)
    for stmt_list in get_statement_lists(func):
        if any(isinstance(stmt, ir.ForLoop) for stmt in stmt_list):
            repl = []
            for stmt in stmt_list:
                if isinstance(stmt, ir.ForLoop):
                    clobbered = preheader_rename_parameters(stmt)
                    if clobbered:
                        renamed = {}
                        if verbose:
                            clobber_str = ', '.join(c.name for c in clobbered if isinstance(c, ir.NameRef))
                            msg = f'The following names appear in the list of loop iterables and are completely ' \
                                  f'overwritten in the loop body: "{clobber_str}"'
                            print(msg)
                        for c in clobbered:
                            if isinstance(c, ir.NameRef):
                                renamed[c] = symbols.make_versioned(c)
                            else:
                                t = typer(c)
                                renamed[c] = symbols.make_unique_name_like('v', t)
                        stmt.iterable = rewrite_expr(renamed, stmt.iterable)
                        pos = stmt.pos
                        for original, rename in renamed.items():
                            copy_stmt = ir.Assign(target=rename, value=original, pos=pos)
                            repl.append(copy_stmt)
                repl.append(stmt)
            stmt_list.clear()
            stmt_list.extend(repl)


def make_min_affine_seq(step, start_and_stops):
    starts = set()
    stops = set()
    for start, stop in start_and_stops:
        starts.add(start)
        stops.add(stop)
    stops.discard(None)
    if len(starts) == 1:
        start = starts.pop()
        if len(stops) == 1:
            stop = stops.pop()
        else:
            stop = ir.MinReduction(*stops)
        return ir.AffineSeq(start, stop, step)
    elif len(stops) == 1:
        stop = stops.pop()
        stop = ir.MAX(ir.SUB(stop, ir.MaxReduction(*starts)), ir.Zero)
        # need a zero here, since
        return ir.AffineSeq(ir.Zero, stop, step)
    else:
        diffs = []
        for start, stop in start_and_stops:
            d = ir.SUB(stop, start)
            d = simplify_arithmetic(d)
            d = fold_constants(d)
            diffs.append(d)
        min_diff = ir.MAX(ir.MinReduction(*diffs), ir.Zero)
        return ir.AffineSeq(ir.Zero, min_diff, step)


def make_affine_seq(node: ir.ValueRef):
    if isinstance(node, ir.AffineSeq):
        return node
    elif isinstance(node, ir.Subscript):
        index = node.index
        value = node.value
        if isinstance(index, ir.Slice):
            start = index.start
            step = index.step
            stop = ir.SingleDimRef(value, ir.Zero)
            if index.stop != ir.SingleDimRef(node.value, ir.Zero):
                stop = ir.MIN(index.stop, stop)
        else:
            start = ir.Zero
            stop = ir.SingleDimRef(value, ir.One)
            step = ir.One
        return ir.AffineSeq(start, stop, step)
    else:
        return ir.AffineSeq(ir.Zero, ir.SingleDimRef(node, ir.Zero), ir.One)


def make_single_index_loop(header: ir.ForLoop, symbols, noescape):
    """

        Make loop interval of the form (start, stop, step).

        This tries to find a safe method of calculation.

        This assumes (with runtime verification if necessary)
        that 'stop - start' will not overflow.

        References:
            LIVINSKII et. al, Random Testing for C and C++ Compilers with YARPGen
            Dietz et. al, Understanding Integer Overflow in C/C++
            Bachmann et. al, Chains of Recurrences - a method to expedite the evaluation of closed-form functions
            https://developercommunity.visualstudio.com/t/please-implement-integer-overflow-detection/409051
            https://numpy.org/doc/stable/user/building.html


        Note: this can complicate range analysis if it forces normalization, so it should be performed late
        # Todo: add reference, think it's a Michael Wolf paper that mentions this

    """

    typer = TypeHelper(symbols)
    normalize = NormalizeMinMax(symbols)

    by_iterable = {}
    target_to_iterable = {}
    for target, iterable in unpack_iterated(header.target, header.iterable):
        interval = make_affine_seq(iterable)
        by_iterable[iterable] = interval
        target_to_iterable[target] = iterable

    by_step = defaultdict(set)
    for interval in by_iterable.values():
        by_step[interval.step].add((interval.start, interval.stop))
    exprs = []
    for step, start_and_stop in by_step.items():
        exprs.append(make_min_affine_seq(step, start_and_stop))

    if len(exprs) == 1:
        # we can avoid normalizing
        affine_expr: ir.AffineSeq = exprs.pop()
    else:
        iterable_lens = []
        for expr in exprs:
            c = compute_element_count(expr.start, expr.stop, expr.step)
            iterable_lens.append(c)

        count = ir.MinReduction(*iterable_lens)
        affine_expr = ir.AffineSeq(ir.Zero, count, ir.One)

    loop_expr = normalize(affine_expr)
    assert isinstance(loop_expr, ir.AffineSeq)
    loop_index = None
    for target in target_to_iterable:
        iterable = target_to_iterable[target]
        if loop_index is None:
            if isinstance(iterable, ir.AffineSeq):
                if iterable.start == loop_expr.start and iterable.step == loop_expr.step and target in noescape:
                    loop_index = target
    # we only use an existing variable as an index if the variable name doesn't escape the loop
    # since the index itself is assigned when testing whether to enter the loop rather than right before entry
    if loop_index is None:
        loop_index = symbols.make_unique_name_like('i', numpy.dtype('int64'))

    body = []
    pos = header.pos
    for target, iterable in unpack_iterated(header.target, header.iterable):
        if target is loop_index:
            continue
        if isinstance(iterable, ir.AffineSeq):
            # we can reuse
            target_type = symbols.check_type(target)
            index_type = symbols.check_type(loop_index)
            if target_type == index_type:
                value = loop_index
            else:
                value = ir.CAST(loop_index, target_type)
        elif isinstance(iterable, ir.NameRef):
            value = ir.Subscript(iterable, loop_index)
        else:
            assert isinstance(iterable, ir.Subscript)
            if isinstance(iterable.index, ir.Slice):
                iterable_index = iterable.index
                if iterable_index.start == loop_expr.start and iterable_index.step == loop_expr.step:
                    value = ir.Subscript(iterable.value, loop_index)
                else:
                    assert loop_expr.step == ir.One and loop_expr.start == ir.Zero
                    if iterable_index.step != ir.One:
                        offset = ir.MULT(loop_index, iterable_index.step)
                        if iterable_index.start != ir.Zero:
                            offset = ir.ADD(offset, iterable_index.start)
                    elif iterable_index.start != ir.Zero:
                        offset = ir.ADD(loop_index, iterable_index.start)
                    else:
                        # slice doesn't augment start or step
                        offset = loop_index
                    value = ir.Subscript(iterable.value, offset)
            else:
                # single index, which means it's double sliced
                t = typer(iterable)
                name = symbols.make_unique_name_like(iterable, t)
                stmt = ir.Assign(name, iterable, pos)
                body.append(stmt)
                assert loop_expr.start == ir.Zero and loop_expr.step == ir.One
                value = ir.Subscript(name, loop_index)
        assign = ir.Assign(target, value, pos)
        body.append(assign)
    body.extend(header.body)
    header = ir.ForLoop(loop_index, loop_expr, body, pos)

    return header


def get_assigned_or_augmented(node: Union[ir.StmtBase, List[ir.StmtBase]]) -> Tuple[Set[ir.NameRef], Set[ir.NameRef]]:
    bound = set()
    augmented = set()
    for stmt in walk_nodes(node):
        if isinstance(stmt, ir.Assign):
            if isinstance(stmt.target, ir.NameRef):
                bound.add(stmt.target)
            elif isinstance(stmt.target, ir.Subscript):
                augmented.add(stmt.target.value)
        elif isinstance(stmt, ir.InPlaceOp):
            if isinstance(stmt.target, ir.NameRef):
                augmented.add(stmt.target)
            else:
                assert isinstance(stmt.target, ir.Subscript)
                if isinstance(stmt.target.value, ir.NameRef):
                    augmented.add(stmt.target.value)
        elif isinstance(stmt, ir.ForLoop):
            # nested loop should not clobber
            for target, _ in unpack_iterated(stmt.target, stmt.iterable):
                if isinstance(target, ir.NameRef):
                    bound.add(target)
    return bound, augmented


def get_safe_loop_indices(stmt: ir.ForLoop, live_on_exit: Set[ir.NameRef], symbols: SymbolTable):
    assigned, augmented = get_assigned_or_augmented(stmt.body)
    # It's technically safe for arrays to be augmented here, but we need to freeze any value
    # used to determine indexing
    assigned.update(augmented)
    assigned.update(live_on_exit)
    possible_indices = [target for (target, iterable) in unpack_iterated(stmt.target, stmt.iterable) if isinstance(iterable, ir.AffineSeq) and target not in assigned]
    # check that on the off chance of multiple assignment, we go by the last assigned
    invalid = set()
    for target, iterable in unpack_iterated(stmt.target, stmt.iterable):
        if not isinstance(iterable, ir.AffineSeq):
            invalid.add(target)
        else:
            invalid.discard(target)
    # Now extract any that are unambiguously typed as integers
    safe_targets = [target for target in possible_indices if is_integer(symbols.check_type(target))]
    return safe_targets


def lower_loops(func: ir.Function, symbols: SymbolTable, verbose: bool = False):
    rename_clobbered_loop_parameters(func, symbols, verbose)
    graph = build_function_graph(func)
    liveness = find_live_in_out(graph)
    loop_blocks = [block for block in graph.nodes() if block.is_loop_block and isinstance(block.first, ir.ForLoop)]
    headers = []
    # our graph is a volatile view rather than a transformable CFG, as this sidesteps issues of maintaining
    # full structure
    safe_target_lists = []
    for block in loop_blocks:
        # grab loop setup statement and any variables that are safe to use as loop indices
        header = block.first
        invalid_iterables = [*invalid_loop_iterables(header, symbols)]
        if invalid_iterables:
            formatter = PrettyFormatter()
            formatted = ', '.join(formatter(iterable) for iterable in invalid_iterables)
            msg = f'Non iterable references: {formatted} in for loop, line: {header.pos.line_begin}.'
            raise CompilerError(msg)
        headers.append(header)
        exit_block = get_loop_exit_block(graph, block)
        live_on_exit = liveness[exit_block].live_in
        safe_targets = get_safe_loop_indices(header, live_on_exit, symbols)
        safe_target_lists.append(safe_targets)

    for header, no_escape in zip(headers, safe_target_lists):
        repl = make_single_index_loop(header, symbols, no_escape)
        header.target = repl.target
        header.iterable = repl.iterable
        header.body.clear()
        header.body.extend(repl.body)
