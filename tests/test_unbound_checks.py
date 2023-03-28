

import lib.ir as ir
from lib.cfg_builder import build_ir_from_func
from lib.liveness import check_for_maybe_unbound_names


# TODO: Need an imported code -> parsed module path

def simple_unbound():
    print(a)


def unbound_one_branch(a, b):
    if a > 0:
        c = a + b
    else:
        print('nothing to c here')
    print(c)


def unbound_on_empty_iterable(a):
    for v in a:
        print(v)
    print(v)


def test_unbound():

    func_ctx = build_ir_from_func(simple_unbound)
    block_to_unbound_map, assigned_after = check_for_maybe_unbound_names(func_ctx)
    # A single block should declare v as unbound
    assert len(block_to_unbound_map) == 1
    block, unbound_vars = block_to_unbound_map.popitem()
    assert unbound_vars == {ir.NameRef('a')}

    func_ctx = build_ir_from_func(unbound_one_branch)
    block_to_unbound_map, assigned_after = check_for_maybe_unbound_names(func_ctx)
    # A single block should declare v as unbound
    assert len(block_to_unbound_map) == 1
    block, unbound_vars = block_to_unbound_map.popitem()
    assert unbound_vars == {ir.NameRef('c')}

    func_ctx = build_ir_from_func(unbound_on_empty_iterable)
    block_to_unbound_map, assigned_after = check_for_maybe_unbound_names(func_ctx)
    # A single block should declare v as unbound
    assert len(block_to_unbound_map) == 1
    block, unbound_vars = block_to_unbound_map.popitem()
    assert unbound_vars == {ir.NameRef('v')}

