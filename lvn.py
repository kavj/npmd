from errors import CompilerError
from functools import singledispatch
from itertools import islice, tee
from typing import Dict, Iterable, Optional, Tuple
import ir

from symbol_table import SymbolTable
from utils import is_entry_point, unpack_iterated
from visitor import sequence_block_intervals, walk, walk_parameters


def binds(node: ir.StmtBase):
    return isinstance(node, (ir.Assign, ir.InPlaceOp, ir.ForLoop))


@singledispatch
def get_target_value_pairs(node):
    msg = f"No method to extract reads and writes from {node}"
    raise NotImplementedError(msg)


@get_target_value_pairs.register
def _(node: ir.StmtBase) -> Tuple[Optional[ir.ValueRef], Optional[ir.ValueRef]]:
    yield


@get_target_value_pairs.register
def _(node: ir.Assign):
    yield node.target, node.value


@get_target_value_pairs.register
def _(node: ir.InPlaceOp):
    yield node.target, node.value


@get_target_value_pairs.register
def _(node: ir.ForLoop):
    for target, iterable in unpack_iterated(node.target, node.iterable):
        yield target, iterable


@singledispatch
def get_referenced(node):
    return None


@get_referenced.register
def _(node: ir.WhileLoop):
    return node.test


@get_referenced.register
def _(node: ir.IfElse):
    return node.test


@get_referenced.register
def _(node: ir.SingleExpr):
    return node.value


@get_referenced.register
def _(node: ir.Return):
    return node.value


def reconstruct_expr(node: ir.Expression, rewrites: Dict[ir.ValueRef, ir.ValueRef]):
    for value in walk(node):
        if isinstance(value, ir.Expression):
            repl = rewrites.get(value)
            if repl is None:
                repl = value.reconstruct(*(rewrites.get(subexpr, subexpr) for subexpr in value.subexprs))
                # map to self if identical
                rewrites = value if repl == value else repl
    return rewrites[node]


def replace_named_subexpressions(node: ir.ValueRef, expr_to_lvn: Dict[ir.ValueRef, ir.NameRef]):
    """
    This is used so that we can
    :param node:
    :param expr_to_lvn:
    :return:
    """
    if not isinstance(node, ir.Expression):
        return node
    name = expr_to_lvn.get(node)
    if name is not None:
        return name
    repls = []
    for subexpr in node.subexprs:
        if not isinstance(subexpr, ir.Expression):
            repls.append(subexpr)
            continue
        name = expr_to_lvn.get(subexpr, subexpr)
        if name is None:
            name_or_expr = replace_named_subexpressions(subexpr, expr_to_lvn)
            repls.append(name_or_expr)
        else:
            repls.append(name)
    if isinstance(node, ir.ADD) and len(repls) == 1:
        print(node)
    return node.reconstruct(*repls)


def replace_expression_parameters(node: ir.ValueRef, name_to_lvn: Dict[ir.NameRef, ir.NameRef]):
    if isinstance(node, ir.Expression):
        subexprs = []
        for subexpr in node.subexprs:
            subexpr_rewrite = replace_expression_parameters(subexpr, name_to_lvn)
            subexprs.append(subexpr_rewrite)
        return node.reconstruct(*subexprs)
    elif isinstance(node, ir.NameRef):
        return name_to_lvn.get(node, node)
    else:
        return node


def rewrite_expression(node: ir.ValueRef,
                       name_to_lvn: Dict[ir.NameRef, ir.NameRef],
                       expr_to_lvn: Dict[ir.ValueRef, ir.NameRef]):
    # First replace all source names with their localized counterparts
    initial_rewrite = replace_expression_parameters(node, name_to_lvn)
    # next replace any explicitly locally bound expression with the name it's bound to
    with_named_exprs = replace_named_subexpressions(initial_rewrite, expr_to_lvn)
    return with_named_exprs


@singledispatch
def rewrite_statement(node,
                      name_to_lvn: Dict[ir.NameRef, ir.NameRef],
                      expr_to_lvn: Dict[ir.ValueRef, ir.NameRef],
                      symbols: SymbolTable):
    raise TypeError


@rewrite_statement.register
def _(stmt: ir.StmtBase,
      src_name_to_lvn: Dict[ir.NameRef, ir.NameRef],
      expr_to_lvn: Dict[ir.ValueRef, ir.NameRef],
      symbols: SymbolTable):
    return stmt


@rewrite_statement.register
def _(stmt: ir.Assign,
      src_name_to_lvn: Dict[ir.NameRef, ir.NameRef],
      expr_to_lvn: Dict[ir.ValueRef, ir.NameRef],
      symbols: SymbolTable):
    value = rewrite_expression(stmt.value, src_name_to_lvn, expr_to_lvn)
    if isinstance(stmt.target, ir.NameRef):
        if stmt.target in src_name_to_lvn:
            target_type = symbols.check_type(stmt.target)
            target_name = symbols.make_unique_name_like(stmt.target, target_type)
            src_name_to_lvn[stmt.target] = target_name
            return ir.Assign(target=target_name, value=value, pos=stmt.pos)
        else:
            return ir.Assign(target=stmt.target, value=value, pos=stmt.pos)
    else:
        # no binding operations
        target = rewrite_expression(stmt.target, src_name_to_lvn, expr_to_lvn)
        return ir.Assign(target=target, value=value, pos=stmt.pos)


@rewrite_statement.register
def _(stmt: ir.InPlaceOp,
      src_name_to_lvn: Dict[ir.NameRef, ir.NameRef],
      expr_to_lvn: Dict[ir.ValueRef, ir.NameRef],
      symbols: SymbolTable):
    value = rewrite_expression(stmt.value, src_name_to_lvn, expr_to_lvn)
    if isinstance(stmt.target, ir.NameRef):
        if stmt.target in src_name_to_lvn:
            target_type = symbols.check_type(stmt.target)
            target_name = symbols.make_unique_name_like(stmt.target, target_type)
            src_name_to_lvn[stmt.target] = target_name
            return ir.InPlaceOp(target=target_name, value=value, pos=stmt.pos)
        else:
            return ir.InPlaceOp(target=stmt.target, value=value, pos=stmt.pos)
    else:
        # no binding operations
        target = rewrite_expression(stmt.target, src_name_to_lvn, expr_to_lvn)
        return ir.InPlaceOp(target=target, value=value, pos=stmt.pos)


@rewrite_statement.register
def _(stmt: ir.SingleExpr,
      name_to_lvn: Dict[ir.NameRef, ir.NameRef],
      expr_to_lvn: Dict[ir.ValueRef, ir.NameRef],
      symbols: SymbolTable):
    if isinstance(stmt.value, ir.Expression):
        expr = rewrite_expression(stmt.value, name_to_lvn, expr_to_lvn)
        return ir.SingleExpr(value=expr, pos=stmt.pos)
    elif isinstance(stmt.value, ir.NameRef):
        name = name_to_lvn.get(stmt.value, stmt.value)
        return ir.SingleExpr(value=name, pos=stmt.pos)
    else:
        return stmt


@rewrite_statement.register
def _(stmt: ir.Return,
      name_to_lvn: Dict[ir.NameRef, ir.NameRef],
      expr_to_lvn: Dict[ir.ValueRef, ir.NameRef],
      symbols: SymbolTable):
    if stmt.value is None:
        return stmt
    value = rewrite_expression(stmt.value, name_to_lvn, expr_to_lvn)
    return ir.Return(value=value, pos=stmt.pos)


def get_single_region_rename_targets(block: Iterable[ir.StmtBase]):
    """
    This checks for a pattern of
    two groups of reads, separated by one or more writes.

    :param block:
    :return:
    """
    war = set()
    read = set()
    must_rename = set()

    for stmt in block:
        if binds(stmt):
            for target, value in get_target_value_pairs(stmt):
                if value is not None:
                    for name in walk_parameters(value):
                        read.add(name)
                        if name in war:
                            must_rename.add(name)
                if isinstance(target, ir.NameRef):
                    if target in read:
                        war.add(target)
                else:
                    for name in walk_parameters(target):
                        read.add(name)
                        if name in war:
                            must_rename.add(name)
        else:
            value = get_referenced(stmt)
            if value is not None:
                for name in walk_parameters(value):
                    read.add(name)
                    if name in war:
                        must_rename.add(name)
    return must_rename


def run_block_lvn(node: Iterable[ir.StmtBase], symbols: SymbolTable, expr_to_lvn: Optional[Dict[ir.Expression, ir.NameRef]] = None):
    repl = []
    if expr_to_lvn is None:
        expr_to_lvn = {}
    # initialize with trivial
    nv_0, nv_1 = tee(node)
    name_to_lvn = {name: name for name in get_single_region_rename_targets(nv_0)}

    for stmt in nv_1:
        stmt = rewrite_statement(stmt, name_to_lvn, expr_to_lvn, symbols)
        repl.append(stmt)

    return repl


def run_local_value_numbering(node: list, symbols: SymbolTable):
    repl = []
    for start, stop in sequence_block_intervals(node):
        if is_entry_point(node[start]):
            entry_point = node[start]
            if isinstance(entry_point, ir.IfElse):
                test = entry_point.test
                if_branch = run_local_value_numbering(entry_point.if_branch, symbols)
                else_branch = run_local_value_numbering(entry_point.else_branch, symbols)
                pos = entry_point.pos
                branch = ir.IfElse(test, if_branch, else_branch, pos)
                repl.append(branch)
            elif isinstance(entry_point, ir.ForLoop):
                target = entry_point.target
                iterable = entry_point.iterable
                body = run_local_value_numbering(entry_point.body, symbols)
                pos = entry_point.pos
                header = ir.ForLoop(target, iterable, body, pos)
                repl.append(header)
            elif isinstance(entry_point, ir.WhileLoop):
                test = entry_point.test
                body = run_local_value_numbering(entry_point.body, symbols)
                pos = entry_point.pos
                header = ir.WhileLoop(test, body, pos)
                repl.append(header)
            else:
                msg = f"Unsupported type {entry_point}"
                raise CompilerError(msg)
        else:
            block_iter = islice(node, start, stop)
            block_repl = run_block_lvn(block_iter, symbols)
            repl.extend(block_repl)
    return repl
