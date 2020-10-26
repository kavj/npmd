import functools
import itertools
import operator

import ir
from visitor import VisitorBase


class Block:
    is_exit_block = False

    def __init__(self, label):
        self.label = label
        self.stmts = []
        self.preds = set()
        self.succs = set()

    def add_stmt(self, stmt):
        self.stmts.append(stmt)

    def add_succ(self, block):
        self.succs.add(block)
        block.preds.add(self)

    def remove_succ(self, block):
        self.succs.remove(block)
        block.preds.remove(self)

    def clear_succs(self):
        for s in self.succs:
            s.preds.remove(self)
        self.preds.clear()

    def clear_preds(self):
        for p in self.preds:
            p.succs.remove(self)
        self.preds.clear()

    def detach(self):
        for p in self.preds:
            if self not in p.succs:
                print(f"{self.label} not in {p.label} successors")
        for s in self.succs:
            if self not in s.preds:
                print(f"{self.label} not in {s.label} predecessors")
        self.clear_preds()
        self.clear_succs()

    @property
    def single_successor(self):
        return len(self.succs) == 1

    @property
    def single_predecessor(self):
        return len(self.preds) == 1

    @property
    def multiple_successors(self):
        return len(self.succs) > 1

    @property
    def multiple_predecessors(self):
        return len(self.preds) > 1

    @property
    def has_successor(self):
        return len(self.succs) > 0

    @property
    def has_predecessor(self):
        return len(self.preds) > 0

    @property
    def successor(self):
        assert (len(self.succs) == 1)
        return next(iter(self.succs))

    @property
    def predecessor(self):
        assert (len(self.succs) == 1)
        return next(iter(self.preds))

    @property
    def empty(self):
        return operator.not_(operator.truth(self.stmts))

    def dump(self):
        return "\n".join(str(stmt) for stmt in self.stmts)

    def __str__(self):
        return str(self.label)

    def __hash__(self):
        return hash(self.label)

    def __bool__(self):
        return operator.truth(self.stmts)

    def __eq__(self, other):
        return self.label == other.label

    def __ne__(self, other):
        return self.label != other.label


class BlockGen:

    def __init__(self, gen=None):
        self.gen = gen if gen is not None else itertools.count()

    def next_block(self) -> Block:
        return Block(next(self.gen))


class ProcedureGraph:
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

    def remove_block(self, block: Block):
        block.detach()
        self.blocks.remove(block)

    def is_empty(self):
        return all(b.isempty for b in self.blocks)

    def dominators(self):
        return compute_doms(self.entry)

    def __bool__(self):
        return operator.truth(self.blocks)


class LoopGraph:
    """ 
    Acyclic and subgraph are used to aid in vectorization. Since we need to vectorize,
    this means renaming of promoted variables.
    Most vectorization methods remove back edges from a removable flow graph, so acyclic view is there. 

    subgraphs are needed sometimes to interact with a vectorizable section. In particular, we are concerned with 
    the initial value and evolution of a particular non-SSA name. Generally this is used to determine whether

    This isn't including builder logic since it may be converted from multiple input types.

    """

    def __init__(self, entry):
        self.entry = entry
        self.blocks = {entry}

    def remove_block(self, block: Block):
        block.detach()
        self.blocks.remove(block)

    def is_empty(self):
        return all(b.isempty for b in self.blocks)

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
            new_idom = functools.reduce(intersect, (m for m in n.preds if m in idoms))
            if n not in idoms or idoms[n] != new_idom:
                idoms[n] = new_idom
                changed = True
    return idoms


class FlowBuilder(VisitorBase):

    def __call__(self, entry):
        if isinstance(entry, ir.Module):
            return [self.visit(f) for f in entry.funcs]
        elif isinstance(entry, ir.Function):
            graph = ProcedureGraph(entry)
            self.hierarchy.append(graph.entry)
            self.visit(entry.body)
        elif entry.is_loop_entry:
            graph = LoopGraph(entry)
        else:
            raise NotImplementedError
        return graph

    def __init__(self):
        self.hierarchy = []
        self.gen = BlockGen()
