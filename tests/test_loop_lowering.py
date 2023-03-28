import inspect
import itertools

import pytest

import lib.ir as ir

from lib.blocks import FunctionContext
from lib.cfg_builder import build_module_ir
from lib.errors import CompilerError
from lib.expression_utils import find_element_count
from lib.graph_walkers import walk_graph
from lib.loop_lowering import lower_loops, rename_clobbered_loop_parameters
from tree_tests import test_indexing
from lib.type_checks import infer_types
from tests.type_info import type_detail
from lib.unpacking import unpack_loop_iter


src = inspect.getfile(test_indexing)
types = type_detail['test_indexing.py']
mod = build_module_ir(src, types)


def get_single_for_loop(func: FunctionContext):
    loop_blocks = [block for block in walk_graph(func.graph) if block.is_loop_block]
    assert len(loop_blocks) == 1
    block = loop_blocks.pop()
    rename_clobbered_loop_parameters(func, block)
    infer_types(func)
    lower_loops(func)
    # There should be a single loop, indexed with variable i
    loops = [stmt for stmt in itertools.chain(*walk_graph(func)) if isinstance(stmt, (ir.ForLoop, ir.WhileLoop))]
    assert len(loops) == 1
    loop = loops.pop()
    assert isinstance(loop, ir.ForLoop)
    return loop


# Todo: This needs to raise if a function tries to iterate over a non-iterable type

def test_simple_index():
    """
    test usable indices using range and enumerate
    :return:
    """
    for j in range(2):
        loop = get_single_for_loop(mod.functions[j])
        assert loop.target == ir.NameRef('i')
        assert isinstance(loop.iterable, ir.AffineSeq)
        assert loop.iterable.start == ir.Zero
        assert loop.iterable.step == ir.One


def test_clobbered_index():
    """
    test clobbered index not used
    :return:
    """
    # Todo: Add test for nested/conditional clobber
    loop = get_single_for_loop(mod.functions[2])
    assert isinstance(loop.target, ir.NameRef)
    # ensure not using original index
    assert loop.target != ir.NameRef('i')
    assert isinstance(loop.iterable, ir.AffineSeq)
    assert loop.iterable.start == ir.Zero
    assert loop.iterable.step == ir.One


def test_no_index_strided_array():
    for func in itertools.islice(mod.functions, 3, 5):
        # find initial loop strides
        headers = [stmt for stmt in itertools.chain(*walk_graph(func.graph)) if isinstance(stmt, (ir.ForLoop, ir.WhileLoop))]
        assert len(headers) == 1
        header = headers.pop()
        assert isinstance(header.iterable, ir.Subscript) and isinstance(header.iterable.index, ir.Slice)
        step = header.iterable.index.step
        loop = get_single_for_loop(func)
        assert isinstance(loop.iterable, ir.AffineSeq)
        loop_step = loop.iterable.step
        assert step == loop_step
        assert isinstance(loop.target, ir.NameRef)


def test_parameter_consolidation():
    # test matching step and matching start and step
    func = mod.functions[5]
    loop = get_single_for_loop(func)
    assert isinstance(loop.iterable, ir.AffineSeq)
    assert loop.iterable.step == ir.NameRef('k')
    assert loop.target == ir.NameRef('t')


def test_non_integral_index_flagged():
    func = mod.functions[6]
    with pytest.raises(CompilerError):
        infer_types(func)


def test_different_step_sizes():
    # Todo: still a work in progress.. but revealed some issues with pretty printing
    with pytest.raises(AssertionError):
        func = mod.functions[7]
        # get initial unpack
        loops = [stmt for stmt in itertools.chain(*walk_graph(func.graph)) if isinstance(stmt, (ir.ForLoop, ir.WhileLoop))]
        assert len(loops) == 1
        loop = loops.pop()
        targets = []
        values = []
        for target, value in unpack_loop_iter(loop):
            targets.append(target)
            values.append(value)
        loop = get_single_for_loop(func)
        # this should have a minimal index
        assert isinstance(loop.target, ir.NameRef)
        a = ir.NameRef('a')
        b = ir.NameRef('b')
        u = ir.NameRef('u')
        v = ir.NameRef('v')
        assert targets == [u, v]
        iterable = loop.iterable
        assert isinstance(iterable, ir.AffineSeq)
        assert iterable.start == ir.Zero
        assert iterable.step == ir.One
        # sliced subscripts
        counts = ir.MIN(*(find_element_count(v) for v in values))

        assert iterable.stop == counts
        assert loop.iterable == ir.AffineSeq(ir.Zero, counts, ir.One)
        assign_u = loop.body[0]
        assign_v = loop.body[1]
        assert assign_u.target == u
        assert assign_v.target == u
        assert isinstance(u, ir.Subscript)
        assert isinstance(v, ir.Subscript)
        # for stmt in zip(pairs, itertools.islice(loop.body, 2)):

