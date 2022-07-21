import itertools
import operator

import networkx as nx

from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Tuple, Union

import ir

from errors import CompilerError
from pretty_printing import PrettyFormatter
from utils import is_entry_point


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
        self.by_statement_list = defaultdict(set)

    def nodes(self):
        return self.graph.nodes()

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

    def reachable_blocks(self) -> Generator[BasicBlock, None, None]:
        entry_block = self.entry_block
        for node, in_degree in self.graph.in_degree():
            if in_degree == 0 and node is not entry_block:
                yield node

    def reachable_blocks_in_list(self, stmts: List[ir.Statement]) -> Generator[BasicBlock, None, None]:
        if stmts is self.entry_block:
            yield stmts
        else:
            for block in self.by_statement_list[id(stmts)]:
                if self.graph.in_degree(block) != 0:
                    yield block

    def get_blocks_in_list(self, block: List[ir.Statement]) -> Generator[BasicBlock, None, None]:
        yield from self.by_statement_list[id(block)]

    def matching_list_and_start(self, stmts: List[ir.Statement], index: int):
        for block in self.by_statement_list[id(stmts)]:
            if block.start == index:
                return block

    def matching_list_and_stop(self, stmts: List[ir.Statement], index: int):
        for block in self.by_statement_list[id(stmts)]:
            if block.stop == index:
                return block

    def first_block(self, stmts: List[ir.Statement]) -> BasicBlock:
        b = self.matching_list_and_start(stmts, 0)
        if b is None:
            msg = f'Graph is incomplete, cannot find a leading block for statement list "{stmts}"'
            raise ValueError(msg)
        return b

    def last_block(self, stmts: List[ir.Statement]) -> BasicBlock:
        b = self.matching_list_and_stop(stmts, len(stmts))
        if b is None:
            # we have to keep dead blocks in the graph to avoid missing them here
            msg = f'Graph is incomplete, cannot find a final block for statement list "{stmts}"'
            raise ValueError(msg)
        return b

    def get_branch_entry_points(self, block: BasicBlock) -> Tuple[BasicBlock, BasicBlock]:
        branch_stmt = block.first
        return self.first_block(branch_stmt.if_branch), self.first_block(branch_stmt.else_branch)

    def get_branch_exit(self, block: BasicBlock) -> Optional[BasicBlock]:
        assert block.is_branch_block
        return self.matching_list_and_start(block.statements, block.stop)

    def get_loop_body(self, block: BasicBlock) -> Optional[BasicBlock]:
        assert block.is_loop_block
        return self.last_block(block.statements)
        # has to be optional to avoid exploding on unreachable loop body, eg while False: ...

    def get_loop_exit(self, block: BasicBlock) -> Optional[BasicBlock]:
        assert block.is_loop_block
        return self.matching_list_and_start(block.statements, block.stop)
        # has to be optional to avoid exploding on unreachable loop body, eg while False: ...


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
        for block in (source, sink):
            assert isinstance(block, BasicBlock)
            if not self.graph.graph.has_node(block):
                self.graph.by_statement_list[id(block.statements)].add(block)
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
        block = BasicBlock(statements, start, stop, builder.next_label, builder.depth)
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


def get_loop_header_block(graph: nx.DiGraph, node: ir.ForLoop) -> BasicBlock:
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


def get_loop_exit_block(graph: nx.DiGraph, node: Union[ir.ForLoop, ir.WhileLoop, BasicBlock]) -> Optional[BasicBlock]:
    if isinstance(node, ir.ForLoop):
        node = get_loop_header_block(graph, node)
    for block in graph.successors(node):
        if block.depth == node.depth:
            return node


def get_blocks_in_loop(graph: FlowGraph, block: BasicBlock, include_header=False):
    # first get the corresponding block
    assert block.is_loop_block
    # Since we have the tree IR intact, we're going to do this by checking corresponding stmt lists
    header = block.first
    blocks = [header] if include_header else []
    queued = deque()
    queued.append(graph.get_blocks_in_list(header.body))
    while queued:
        for block in queued[-1]:
            blocks.append(block)
            # Todo: Do we need to distinguish nested loops?
            if block.is_entry_point:
                queued.appendleft(graph.successors(block))
        queued.pop()
    return blocks


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
    stmts = block.statements
    assert block.stop == len(stmts)
    # Todo: need a good replace block for minor stuff like this
    # get positional info
    if block:
        line = block.last.pos.line_begin
        pos = ir.Position(line, line, 0, 10)
    else:
        pos = ir.Position(-1, -1, 0, 10)
    stmt = ir.Return(ir.NoneRef(), pos)
    stmts.append(stmt)
    preds = graph.predecessors(block)
    graph.graph.remove_node(block)
    updated = BasicBlock(block.statements, block.start, block.stop + 1, block.label, block.depth)
    for p in preds:
        graph.graph.add_edge(p, updated)


def graph_to_pydot(graph: nx.DiGraph, dot_name: str, module_name: str = None, path: Optional[Path] = None):
    """
    simple func to generate a dot file
    :param graph:
    :param dot_name:
    :param module_name:
    :param path:
    :return:
    """
    # find entry node
    assert dot_name is not None
    if module_name is not None:
        dot_name = f'{module_name}_{dot_name}'
    dot_name = Path(dot_name).with_suffix('.dot')
    if path is None:
        dot_path = Path(dot_name)
    else:
        dot_path = path.joinpath(dot_name)

    nx.drawing.nx_pydot.write_dot(graph.graph, dot_path)
    return dot_path


def render_pydot(dot_file: Path, out_path: Path):
    import pydot
    graph = pydot.graph_from_dot_file(dot_file)
    graph = graph[0]
    img_name = f'{dot_file.name}.png'
    render_path = out_path.joinpath(img_name)
    graph.write_png(render_path)
