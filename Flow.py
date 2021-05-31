import itertools
import operator

from functools import reduce, singledispatchmethod

import ir
from visitor import VisitorBase


class Block:
    block_labeler = None
    successors = None
    stmts = None
    label = None

    @property
    def single_successor(self):
        return len(self.successors) == 1

    @property
    def multiple_successors(self):
        return len(self.successors) > 1

    @property
    def has_successor(self):
        return len(self.successors) > 0

    @property
    def successor(self):
        assert (len(self.successors) == 1)
        return next(iter(self.successors))

    @property
    def empty(self):
        return operator.not_(operator.truth(self.stmts))

    def dump(self):
        return "\n".join(str(stmt) for stmt in self.stmts)

    def __str__(self):
        return str(self.label)

    def __hash__(self):
        return hash(self.label)


class BasicBlock(Block):

    def __init__(self, stmts=None):
        self.label = next(self.block_labeler)
        self.stmts = stmts
        self.successors = set()

    def add_stmt(self, stmt):
        is_branch_point = isinstance(stmt, (ir.ForLoop, ir.IfElse, ir.WhileLoop))
        if self.stmts:
            if is_branch_point:
                raise ValueError
            self.stmts.append(stmt)
        elif is_branch_point:
            self.stmts = stmt
        else:
            self.stmts = [stmt]

    @property
    def terminated(self):
        return len(self.stmts) > 0 and isinstance(self.stmts[-1], (ir.Break, ir.Return))


class LoopHeader(Block):

    def __init__(self, header):
        self.stmts = header
        self.label = next(self.block_labeler)
        self.successors = set()
        self.latch_block = None
        self.post_block = None


class BranchHeader(Block):

    def __init__(self, header):
        self.stmts = header
        self.label = next(self.block_labeler)
        self.if_block = None
        self.else_block = None
        self.post_block = None


class FuncGraph:
    """
    Acyclic and subgraph are used to aid in vectorization. Since we need to vectorize,
    this means renaming of promoted variables.
    Most vectorization methods remove back edges from a removable flow graph, so acyclic view is there.

    subgraphs are needed sometimes to interact with a vectorizable section. In particular, we are concerned with
    the initial value and evolution of a particular non-SSA name. Generally this is used to determine whether

    This isn't including builder logic since it may be converted from multiple input types.

    """

    def __init__(self, entry):
        # this should auto-block wrap if entry is not a block
        self.entry = entry
        self.blocks = {entry}

    def empty(self):
        return all(b.empty for b in self.blocks)

    def dominators(self):
        return compute_doms(self.entry)

    def __bool__(self):
        return operator.truth(self.blocks)


# The following are based on
# Keith D. Cooper et. al., A Simple, Fast Dominance Algorithm


def find_entry_points(blocks):
    entry_points = set()
    for node in blocks:
        if not node.preds:
            entry_points.add(node)
    return entry_points


def post_order_nodes(blocks):
    entry = find_entry_points(blocks)
    if len(entry) != 1:
        raise ValueError("multiple entry points")
    entry = entry.pop()

    visited = {entry}
    queued = [(entry, iter(entry.succs))]
    order = []

    while queued:
        try:
            node = next(queued[-1][1])
            if node not in visited and node in blocks:
                visited.add(node)
                print("adding: ", node)
                queued.append((node, iter(node.succs)))
        except StopIteration:
            node, _ = queued.pop()
            order.append(node)
    return order


# The following are based on
# Keith D. Cooper et. al., A Simple, Fast Dominance Algorithm


def compute_doms(blocks):
    ordered = post_order_nodes(blocks)
    entry = ordered[-1]
    idoms = {entry: entry}
    postorder = {n: index for index, n in enumerate(ordered)}
    ordered.pop()
    ordered.reverse()

    def intersect(n, m):
        while n != m:
            while postorder[n] < postorder[m]:
                n = idoms[n]
            while postorder[n] > postorder[m]:
                m = idoms[m]
        return n

    changed = True
    while changed:
        changed = False
        for n in ordered:
            new_idom = reduce(intersect, (m for m in n.preds if m in idoms))
            if n not in idoms or idoms[n] != new_idom:
                idoms[n] = new_idom
                changed = True
    return idoms


class FlowBuilder(VisitorBase):

    def __init__(self):
        self.post_loop_blocks = []
        self.headers = []
        self.exit_block = None
        self.current_block = None
        self.blocks = None

    def __call__(self, entry):
        assert (isinstance(entry, ir.Function))
        BasicBlock.block_labeler = itertools.count()
        self.blocks = set()
        self.entry = entry
        self.visit(entry)

    def add_basic_block(self, stmts=None):
        bb = BasicBlock(stmts)
        self.blocks.add(bb)
        return bb

    def update_current_block(self, block=None):
        if block is None:
            block = self.add_basic_block()
        self.current_block = block
        return block

    def add_predecessor(self, block):
        block.successors.add(self.current_block)

    def add_successor(self, block):
        self.current_block.successors.add(block)

    def push_loop_entry(self, header):
        # wrap header
        current = self.current_block
        self.update_current_block(header)
        self.add_predecessor(current)
        post = self.add_basic_block()
        self.add_successor(post)
        self.headers.append(header)
        self.post_loop_blocks.append(post)
        # create loop body

    def push_loop_exit(self):
        # This assumes fake loops are already rewritten
        assert not self.current_block.terminated
        header = self.headers.pop()
        self.add_successor(header)
        post = self.post_loop_blocks.pop()
        self.update_current_block(post)

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Function):
        assert(node is self.entry)
        self.exit_block = self.add_basic_block()
        func_entry = self.add_basic_block(node)
        # returns the first block
        entry_block = self.add_basic_block()
        func_entry.successors.add(entry_block)
        self.current_block = entry_block
        self.visit(node.body)
        if not self.current_block.terminated:
            self.current_block.successors.add(self.exit_block)
        # generate graph
        return func_entry

    @visit.register
    def _(self, node: ir.StmtBase):
        self.current_block.append(node)

    @visit.register
    def _(self, node: ir.ForLoop):
        self.push_loop_entry(node)
        self.visit(node.body)
        self.push_loop_exit()

    @visit.register
    def _(self, node: ir.WhileLoop):
        self.push_loop_entry(node)
        self.visit(node.body)
        self.push_loop_exit()

    @visit.register
    def _(self, node: ir.IfElse):
        prev = self.current_block
        self.update_current_block()
        self.add_predecessor(prev)
        if_block = self.add_basic_block()
        else_block = self.add_basic_block()
        # both sides terminated is really rare
        # which is the only way this becomes an orphan
        post_block = self.add_basic_block()
        self.add_successor(if_block)
        self.add_successor(else_block)
        self.update_current_block(if_block)
        self.visit(node.if_branch)
        if not self.current_block.terminated:
            self.add_successor(post_block)
        self.update_current_block(else_block)
        self.visit(node.else_branch)
        if not self.current_block.terminated:
            self.add_successor(post_block)
        self.update_current_block(post_block)

    @visit.register
    def _(self, node: ir.StmtBase):
        self.current_block.stmts.append(node)

    @visit.register
    def _(self, node: ir.Continue):
        raise ValueError

    @visit.register
    def _(self, node: ir.Break):
        self.current_block.stmts.append(node)
        self.add_successor(self.post_loop_blocks[-1])

    @visit.register
    def _(self, node: ir.Return):
        self.current_block.stmts.append(node)
        self.add_successor(self.exit_block)

    @visit.register
    def _(self, node: list):
        bb = BasicBlock([])
        for stmt in node:
            pass
