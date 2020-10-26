from functools import singledispatchmethod, cached_property

import ir
from visitor import VisitorBase, walk_assignments


class LoopHeaderChecker:
    """
    A general data structure for loop header analysis.

    This can process basically any for loop header.
    It maintains a serialized view of the unpacking without
    any assumptions that this unpacks correctly or that 
    assignments are independent.

    The simple and typical case can be specialized further.

    """

    def __init__(self, header: ir.ForLoop):
        self.serialized = [*walk_assignments(header)]

    @cached_property
    def iterables(self):
        return set(v for (_, v) in self.serialized)

    @cached_property
    def targets(self):
        return set(v for (v, _) in self.serialized)

    @cached_property
    def unpacking_errors(self):
        return {target: value for (target, value) in self.serialized if isinstance(target, ir.Tuple)}

    @cached_property
    def tuple_valued_targets(self):
        return {target: value for (target, value) in self.serialized if isinstance(value, ir.Zip)}

    @cached_property
    def complex_assignments(self):
        return {target: value for (target, value) in self.serialized if not isinstance(target, ir.NameRef)}

    @cached_property
    def is_well_formed_header(self):
        if self.unpacking_errors:
            # in pure python, this would either be an unpacking error or
            # unpacking a named iterator that outputs a sequence of tuples.
            return False
        elif self.complex_assignments:
            return False
        elif self.tuple_valued_targets:
            return False
        return True

    @cached_property
    def is_range_loop(self):
        if len(self.serialized) != 1:
            return False
        _, v = self.serialized[0]
        return isinstance(v, ir.Counter)

    @property
    def as_target_map(self):
        """
        Get final syntactical mapping, ignoring any intermediate dependencies.

        """

        return {target: iterable for (target, iterable) in self.serialized}


class LoopDesc:
    """
    This is the start of a high level loop descriptor.

    It is intended to expose structural information.

    It has to track non-trivial divergent statements (anything beyond if else),
    nesting,

    """

    def __init__(self, is_for_loop):
        self.is_for_loop = is_for_loop
        self.header_map = {}
        self.body_assigns = set()
        self.nested_loops = []
        self.divergent_stmts = []

    def post_order_nested(self):
        for n in self.nested_loops:
            yield from n.post_order_nested()
        yield self

    @property
    def loop_counters(self):
        return {t: v for (t, v) in self.header_map.items() if v.is_counter}

    @property
    def is_while_loop(self):
        return not self.is_for_loop

    @property
    def is_range_loop(self):
        raise NotImplementedError

    @property
    def nested_count(self):
        return len(self.nested_loops)

    @property
    def is_innermost_loop(self):
        return len(self.nested_loops) == 0

    @property
    def nest_depth(self):
        if self.is_innermost_loop:
            return 0
        return 1 + max(loop.nest_depth for loop in self.nested_loops)

    @property
    def may_exit_from_body(self):
        contains_break_or_return = (self.divergent_stmts
                                    and any(not isinstance(stmt, ir.Continue) for stmt in self.divergent_stmts))
        return contains_break_or_return


class LoopNestBuilder(VisitorBase):
    """
    This checks loop divergence, loop counters, nesting structure, 
    and what targets are clobbered in the body of each loop.

    """

    def __init__(self):
        self.loops = []
        self.completed = None

    def __call__(self, node):
        if self.loops:
            raise RecursionError("LoopExitCollector cannot be called recursively")
        elif not node.is_loop_entry:
            raise TypeError
        self.completed = []
        self.visit(node)
        assert (not self.loops)
        completed = self.completed
        self.completed = None
        return completed

    def build_loop_descriptor(self, node):
        if isinstance(node, ir.ForLoop):
            # counters = {target: value for (target, value) in targetmap.items() if value.is_counter}
            loop = LoopDesc(True)
            for target, value in walk_assignments(node):
                loop.header_map[target] = value
        else:
            loop = LoopDesc(False)
        if self.loops:
            self.loops[-1].nested.append(node)
        self.loops.append(loop)
        self.visit(node.body)
        loop = self.loops.pop()
        assert (loop.header is node)
        # only directly track top level
        if not self.loops:
            self.completed.append(loop)

    @singledispatchmethod
    def visit(self, node):
        return super().visit(node)

    @visit.register
    def _(self, node: ir.Assign):
        for target, value in walk_assignments(node):
            if not target.is_subscript:
                self.loops[-1].body_assigns.add(target)

    @visit.register
    def _(self, node: ir.CascadeAssign):
        for target in node.targets:
            if not target.is_subscript:
                self.loops[-1].body_assigns.add(target)

    @visit.register
    def _(self, node: ir.Return):
        if self.loops:
            self.loops[-1].divergent_stmts.append(node)

    @visit.register
    def _(self, node: ir.Break):
        self.loops[-1].divergent_stmts.append(node)

    @visit.register
    def _(self, node: ir.Continue):
        self.loops[-1].divergent_stmts.append(node)

    @visit.register
    def _(self, node: ir.WhileLoop):
        self.build_loop_descriptor(node)

    @visit.register
    def _(self, node: ir.ForLoop):
        self.build_loop_descriptor(node)
