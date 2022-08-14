import itertools
import operator

import networkx as nx

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import npmd.ir as ir

from npmd.errors import CompilerError
from npmd.pretty_printing import PrettyFormatter
from npmd.utils import is_entry_point


formatter = PrettyFormatter()


def get_entry_point(graph: nx.DiGraph):
    for node, degree in graph.in_degree():
        if degree == 0 and isinstance(node.first, ir.Function):
            return node


def matches_while_true(node: ir.Statement):
    if isinstance(node, ir.WhileLoop):
        if isinstance(node.test, ir.CONSTANT) and operator.truth(node.test):
            return True
    return False


@dataclass(frozen=True)
class BasicBlock:
    statements: Union[List[ir.Statement], List[ir.Function]]
    start: int
    stop: int
    label: int  # useful in case going by statement is too verbose
    depth: int

    def __post_init__(self):
        assert isinstance(self.statements, list)
        assert 0 <= self.start <= self.stop <= len(self.statements)

    @property
    def first(self) -> Union[Optional[ir.Statement], Optional[ir.Function]]:
        if self.start != self.stop:
            return self.statements[self.start]

    @property
    def last(self) -> Union[Optional[ir.Statement], Optional[ir.Function]]:
        if self.start != self.stop:
            return self.statements[self.stop-1]

    @property
    def is_entry_block(self):
        return isinstance(self.first, ir.Function)

    @property
    def is_loop_block(self):
        return isinstance(self.first, (ir.ForLoop, ir.WhileLoop))

    @property
    def is_branch_block(self):
        return isinstance(self.first, ir.IfElse)

    @property
    def is_terminated(self):
        return isinstance(self.last, (ir.Break, ir.Continue, ir.Return))

    @property
    def unterminated(self):
        return not self.is_terminated

    @property
    def is_entry_point(self):
        return self.is_loop_block or self.is_branch_block or self.is_entry_block

    @property
    def list_id(self):
        return id(self.statements)

    def append_to_block(self, stmt: ir.StmtBase):
        assert isinstance(stmt, ir.StmtBase)
        if self.stop != len(self.statements):
            msg = f'Cannot append to block that does not terminate the statement list'
            raise CompilerError(msg)
        updated_len = self.stop + 1
        self.statements.append(stmt)
        object.__setattr__(self, 'stop', updated_len)

    def __bool__(self):
        return self.start < self.stop

    def __len__(self):
        return self.stop - self.start

    def __iter__(self):
        return itertools.islice(self.statements, self.start, self.stop)

    def __reversed__(self):
        return reversed(list(iter(self)))

    def __str__(self):
        if self:
            first = self.first
            if isinstance(first, ir.Function):
                formatted = formatter(self.first, truncate_after=20)
                return f'{formatted}\nindex={self.label}\ndepth={self.depth}'
            else:
                assert isinstance(first, ir.StmtBase)
                pos = self.first.pos.line_begin
                formatted = formatter(self.first, truncate_after=20)
                return f'{formatted}\nindex={self.label}\ndepth={self.depth}\nline {pos}'
        else:
            return f'index={self.label}\ndepth={self.depth}'

    def __hash__(self):
        return hash(self.label)
        # return hash((id(self.statements), self.start, self.stop, self.label, self.depth))


class FlowGraph:

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entry_block = None

    @property
    def func_name(self):
        return self.entry_block.first.name

    def nodes(self):
        return self.graph.nodes()

    def reachable_nodes(self):
        return nx.dfs_preorder_nodes(self.graph, self.entry_block)

    def walk_nodes(self):
        for block in self.graph.nodes():
            for stmt in block:
                yield stmt

    def predecessors(self, node: BasicBlock):
        return self.graph.predecessors(node)

    def successors(self, node: BasicBlock):
        return self.graph.successors(node)

    def in_degree(self, block: Optional[BasicBlock] = None):
        if block is None:
            return self.graph.in_degree()
        return self.graph.in_degree(block)

    def out_degree(self, block: Optional[BasicBlock] = None):
        if block is None:
            return self.graph.out_degree()
        return self.graph.out_degree(block)

    def remove_edge(self, source: BasicBlock, sink: BasicBlock):
        self.graph.remove_edge(source, sink)

    def get_branch_entry_points(self, block: BasicBlock) -> Tuple[Optional[BasicBlock], Optional[BasicBlock]]:
        branch_stmt = block.first
        entry_points = [*self.successors(block)]
        if entry_points:
            if len(entry_points) == 1:
                entry_point = entry_points[0]
                if entry_point.statements is branch_stmt.if_branch:
                    return entry_point, None
                else:
                    assert entry_point.statements is branch_stmt.else_branch
                    return None, entry_point
            elif len(entry_points) == 2:
                if entry_points[0].statements is branch_stmt.if_branch and entry_points[1].statements is branch_stmt.else_branch:
                    return tuple(entry_points)
                elif entry_points[1].statements is branch_stmt.else_branch and entry_points[0].statements is branch_stmt.else_branch:
                    return entry_points[1], entry_points[0]
        msg = f'Unable to split branch at line: {block.first.pos.line_begin}'
        raise CompilerError(msg)



def match_block_to_entry_point(graph: FlowGraph, node: Union[ir.Function, ir.ForLoop, ir.WhileLoop, ir.IfElse]):
    if isinstance(node, ir.Function):
        entry_point = graph.entry_block
        assert entry_point.first is node
        return entry_point
    else:
        for block in graph.nodes():
            if block.is_entry_point:
                if block.first is node:
                    return block
    msg = f'Statement "{node}" is not in graph'
    raise ValueError(msg)


def sequence_block_intervals(stmts: Iterable[ir.Statement]):
    """
    For reversal, cast to list and reverse.
    This will intentionally yield a blank interval at the end and between any 2 consecutive scope points.
    :param stmts:
    :return:
    """

    block_start = 0
    block_last = -1
    for block_last, stmt in enumerate(stmts):
        if is_entry_point(stmt):
            if block_start < block_last:
                yield block_start, block_last
            next_start = block_last + 1
            yield block_last, next_start
            block_start = next_start
    # always add an end block. Either it's non-empty or we need a reconvergence block
    yield block_start, block_last + 1


class CFGBuilder:

    def __init__(self, start_from=0):
        self.loop_entry_points = []
        self.continue_map = defaultdict(list)
        self.break_map = defaultdict(list)
        self.scope_entry_blocks = {}  # map of header to initial entry blocks
        self.scope_exit_blocks = {}  # map of header to exit
        self.return_blocks = []
        self.labeler = itertools.count(start_from)
        self.graph = FlowGraph()
        self.counter = itertools.count()

    @property
    def entry_block(self):
        return self.graph.entry_block

    @property
    def next_label(self):
        return next(self.counter)

    @contextmanager
    def enclosing_loop(self, node: BasicBlock):
        assert node.is_loop_block
        self.loop_entry_points.append(node)
        yield
        self.loop_entry_points.pop()

    @property
    def depth(self):
        return len(self.loop_entry_points)

    def create_block(self, stmts: List[ir.StmtBase], start: int, stop: int, depth: int):
        label = next(self.labeler)
        block = BasicBlock(stmts, start, stop, label, depth)
        self.graph.graph.add_node(block)
        return block

    def insert_entry_block(self, func: ir.Function):
        assert self.entry_block is None
        label = next(self.labeler)
        block = BasicBlock([func], 0, 1, label, 0)
        self.graph.graph.add_node(block)
        self.graph.entry_block = block

    def add_edge(self, source: BasicBlock, sink: BasicBlock):
        if not isinstance(source, BasicBlock) or not isinstance(sink, BasicBlock):
            msg = f'Expected BasicBlock type for edge endpoints. Received "{source}" and "{sink}"'
            raise ValueError(msg)
        assert source is not sink
        self.graph.graph.add_edge(source, sink)

    def register_scope_entry_point(self, source: BasicBlock, sink: BasicBlock):
        assert source.is_entry_point
        self.add_edge(source, sink)
        self.scope_entry_blocks[source].append(sink)

    def register_scope_exit_point(self, source: BasicBlock, sink: BasicBlock):
        assert source.is_branch_block or source.is_loop_block
        assert source not in self.scope_exit_blocks
        self.scope_exit_blocks[source] = sink

    def register_continue(self, block: BasicBlock):
        if not self.loop_entry_points:
            msg = f'Break: "{block.last}" encountered outside of loop.'
            raise CompilerError(msg)
        self.continue_map[self.loop_entry_points[-1]].append(block)

    def register_break(self, block: BasicBlock):
        if not self.loop_entry_points:
            msg = f'Break: "{block.last}" encountered outside of loop.'
            raise CompilerError(msg)
        self.break_map[self.loop_entry_points[-1]].append(block)

    def register_block_terminator(self, block: BasicBlock):
        last = block.last
        if last is not None:
            if isinstance(last, ir.Continue):
                self.register_continue(block)
            elif isinstance(last, ir.Break):
                self.register_break(block)
            elif isinstance(last, ir.Return):
                self.return_blocks.append(block)


def build_graph_recursive(statements: List[ir.Statement], builder: CFGBuilder, entry_point: BasicBlock):
    prior_block = entry_point
    deferrals = []  # last_block determines if we have deferrals to this one
    for start, stop in sequence_block_intervals(statements):
        block = builder.create_block(statements, start, stop, builder.depth)
        for stmt in block:
            if stmt.pos.line_begin == 23:
                print('')
        if prior_block is entry_point:
            builder.add_edge(entry_point, block)
        elif prior_block.is_branch_block:
            # If we have blocks exiting a branch, which do not contain a terminating statement
            # then add incoming edges to this block
            if deferrals:
                # mark this as a branch convergence point
                prior_block.first.branch_converge = block
            for d in deferrals.pop():
                builder.add_edge(d, block)
        else:
            if prior_block.is_loop_block:
                # indicate loop exit block so that breaks can be connected
                builder.register_scope_exit_point(prior_block, block)
            # loop or normal must add edge
            if prior_block.unterminated and not matches_while_true(prior_block.last):
                builder.add_edge(prior_block, block)
        # buffering taken care of by sequence block
        if block.is_loop_block:
            # need to preserve entry point info here..
            loop_header_stmt = statements[start]
            with builder.enclosing_loop(block):
                body = loop_header_stmt.body
                last_interior_block = build_graph_recursive(body, builder, block)
                if last_interior_block.unterminated:
                    builder.add_edge(last_interior_block, block)
        elif block.is_branch_block:
            branch_exit_points = []
            if_stmt = statements[start]
            if_body = if_stmt.if_branch
            else_body = if_stmt.else_branch
            if_exit_block = build_graph_recursive(if_body, builder, block)
            # patch initial entry point
            branch_stmt = block.first
            if_entry_block, = builder.graph.successors(block)
            branch_stmt.branch_true = if_entry_block
            if if_exit_block.unterminated:
                branch_exit_points.append(if_exit_block)
            else_exit_block = build_graph_recursive(else_body, builder, block)
            for s in builder.graph.successors(block):
                if s is not if_entry_block:
                    branch_stmt.branch_false = s
                    break
            else:
                msg = f'No else entry block found for {statements[start]}'
                raise ValueError(msg)
            if else_exit_block.unterminated:
                branch_exit_points.append(else_exit_block)
            deferrals.append(branch_exit_points)
        elif block.is_terminated:
            builder.register_block_terminator(block)
        prior_block = block
    return prior_block


def build_function_graph(func: ir.Function) -> FlowGraph:
    builder = CFGBuilder()
    builder.insert_entry_block(func)
    build_graph_recursive(func.body, builder, builder.entry_block)

    # Now clean up the graph
    for loop_header, continue_blocks in builder.continue_map.items():
        for block in continue_blocks:
            builder.add_edge(block, loop_header)

    for loop_header, break_blocks in builder.break_map.items():
        loop_exit_block = builder.scope_exit_blocks[loop_header]
        for block in break_blocks:
            builder.add_edge(block, loop_exit_block)

    return builder.graph


def get_loop_header_block(graph: FlowGraph, node: ir.ForLoop) -> BasicBlock:
    header = None
    # find block matching this header
    for block in graph.nodes():
        if block.is_loop_block:
            if block.first is node:
                header = block
                break
    if header is None:
        msg = f'Loop header {node} was not found in the graph.'
        raise ValueError(msg)
    return header


def get_loop_exit_block(graph: FlowGraph, node: BasicBlock) -> Optional[BasicBlock]:
    for block in graph.successors(node):
        if block.depth == node.depth:
            return block


def dominator_tree(graph: FlowGraph):
    idoms = nx.dominance.immediate_dominators(graph.graph, graph.entry_block)
    # remove self dominance
    g = nx.DiGraph()
    for k, v in idoms.items():
        if k.label != v.label:
            g.add_edge(v, k)
    return g


def get_blocks_in_loop(graph: FlowGraph, block: BasicBlock):
    """
    :param graph:
    :param block:
    :param include_header:
    :return:
    """

    assert block.is_loop_block
    # get the exit block if there is one
    exit_block = get_loop_exit_block(graph, block)
    assert exit_block is not block
    cfg = graph.graph
    if exit_block is not None:
        # make a disconnected sub-graph so that we don't cross divergent edges
        nodes = set(cfg.nodes())
        nodes.remove(exit_block)
        cfg = nx.induced_subgraph(cfg, nodes)

    loop_blocks = set(nx.descendants(cfg, block))

    return loop_blocks


def patch_missing_return(graph: FlowGraph):
    missing = []
    for node, degree in graph.out_degree():
        if degree == 0:
            if node:
                if not isinstance(node.last, ir.Return):
                    missing.append(node)
            else:
                missing.append(node)
    # now extract statmenet lists, should be only one
    if not missing:
        return
    if len(missing) > 1:
        msg = f'Multiple unterminated exit blocks.. somehow..'
        raise ValueError(msg)
    block = missing.pop()
    # verify that this is the terminal block on this statement list
    if block:
        line = block.last.pos.line_begin
        pos = ir.Position(line, line, 0, 10)
    else:
        pos = ir.Position(-1, -1, 0, 10)
    stmt = ir.Return(ir.NoneRef(), pos)
    # Todo: do we need to switch to deque for appendleft?
    block.append_to_block(stmt)


def render_dot_graph(graph: nx.DiGraph, name: str, out_path: Path):
    dot_graph = nx.drawing.nx_pydot.to_pydot(graph)
    img_name = f'{name}.png'
    render_path = out_path.joinpath(img_name)
    dot_graph.write_png(render_path)


def render_dominator_tree(graph: FlowGraph, out_path: Path):
    """
    convenience method to render a dominator tree
    :param graph:
    :param out_path:
    :return:
    """
    dom_tree = dominator_tree(graph)
    img_name = f'{graph.func_name}_doms'
    render_dot_graph(dom_tree, img_name, out_path)
