import itertools
import operator

import networkx as nx

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import lib.ir as ir

from lib.formatting import PrettyFormatter
from lib.symbol_table import SymbolTable

formatter = PrettyFormatter()


@dataclass
class BasicBlock:
    statements: Union[List[ir.Statement], List[ir.Function]]
    label: int  # useful in case going by statement is too verbose
    depth: int

    def replace_statement(self, stmt: ir.Statement, index: int):
        assert 0 <= index < len(self.statements)
        self.statements[index] = stmt

    def append_statement(self, stmt: ir.Statement):
        if self.unterminated:
            # we don't want to add statements past any terminal statement, since this may
            self.statements.append(stmt)

    def pop_statement(self):
        return self.statements.pop()

    def replace_statements(self, stmts: List[ir.Statement]):
        if self.statements:
            if self.is_entry_point:
                assert len(stmts) == 1
                assert isinstance(stmts[0], type(self.first))
            # maintain same object
            self.statements.clear()
            self.statements.extend(stmts)

    @property
    def first(self) -> Union[Optional[ir.Statement], Optional[ir.Function]]:
        if self.statements:
            return self.statements[0]

    @property
    def last(self) -> Union[Optional[ir.Statement], Optional[ir.Function]]:
        if self.statements:
            return self.statements[-1]

    @property
    def is_function_entry(self):
        return isinstance(self.first, ir.Function)

    @property
    def is_loop_block(self):
        return isinstance(self.first, (ir.ForLoop, ir.WhileLoop))

    @property
    def is_branch_block(self):
        return isinstance(self.first, ir.IfElse)

    @property
    def break_terminated(self):
        return isinstance(self.last, ir.Break)

    @property
    def terminated(self):
        return isinstance(self.last, (ir.Break, ir.Continue, ir.Return))

    @property
    def unterminated(self):
        return not self.terminated

    @property
    def is_entry_point(self):
        return self.is_loop_block or self.is_branch_block or self.is_function_entry

    def __bool__(self):
        return operator.truth(self.statements)

    def __len__(self):
        return len(self.statements)

    def __iter__(self):
        return iter(self.statements)

    def __reversed__(self):
        return reversed(self.statements)

    def __str__(self):
        if self:
            first = self.first
            if isinstance(first, ir.Function):
                formatted = formatter(self.first, truncate_after=20)
                return f'{formatted}\nindex={self.label}\ndepth={self.depth}'
            else:
                assert isinstance(first, ir.StmtBase)
                pos = first.pos.line_begin
                formatted = formatter(first, truncate_after=20)
                return f'{formatted}\nindex={self.label}\ndepth={self.depth}\nline {pos}'
        else:
            return f'index={self.label}\ndepth={self.depth}'

    def __hash__(self):
        return hash(self.label)


@dataclass
class FunctionContext:
    graph: nx.DiGraph
    entry_point: BasicBlock
    symbols: Optional[SymbolTable] = None
    labeler: Optional[itertools.count] = None

    def __post_init__(self):
        assert self.entry_point.is_function_entry
        if self.labeler is None:
            self.labeler = itertools.count()

    @cached_property
    def name(self):
        return self.entry_point.first.name

    @property
    def args(self):
        return self.entry_point.first.args

    def add_block(self, statements: List[ir.Statement], depth, parents: Iterable[BasicBlock] = (), children: Iterable[BasicBlock] = ()):
        label = next(self.labeler)
        block = BasicBlock(statements, label, depth)
        self.graph.add_node(block)
        for p in parents:
            self.graph.add_edge(p, block)
        for c in children:
            self.graph.add_edge(block, c)
        return block


def is_loop_entry_block(graph: nx.DiGraph, block: BasicBlock) -> bool:
    if block.depth == 0 or graph.in_degree[block] != 1:
        return False
    pred, = graph.predecessors(block)
    return pred.is_loop_block and pred.depth == block.depth - 1


def make_temporary_assign(func: FunctionContext, base_name: Union[str, ir.NameRef], value: ir.ValueRef, pos: ir.Position):
    if isinstance(base_name, ir.NameRef):
        base_name = base_name.name
    name = func.symbols.make_versioned(base_name)
    assign = ir.Assign(name, value, pos)
    return assign


def dominator_tree(func: FunctionContext) -> nx.DiGraph:
    idoms = nx.dominance.immediate_dominators(func.graph, func.entry_point)
    # remove self dominance
    g = nx.DiGraph()
    for k, v in idoms.items():
        if k.label != v.label:
            g.add_edge(v, k)
    return g


def render_dot_graph(graph: nx.DiGraph, name: str, out_path: Path):
    dot_graph = nx.drawing.nx_pydot.to_pydot(graph)
    img_name = f'{name}.png'
    render_path = out_path.joinpath(img_name)
    dot_graph.write_png(render_path)


def render_dominator_tree(func: FunctionContext, out_path: Path):
    """
    convenience method to render a dominator tree
    :param func:
    :param out_path:
    :return:
    """
    dom_tree = dominator_tree(func)
    img_name = f'{func.name}_doms'
    render_dot_graph(dom_tree, img_name, out_path)
