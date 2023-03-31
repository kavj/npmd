import lib.ir as ir
from lib.walkers import walk_expr, walk_parameters
# check for post ordering


def test_simple_walk():
    """
    Note: u + v is handled as a commutative expression for simpler comparison
    This behavior may not hold in case non-commutative things arise.

    :return:
    """
    a = ir.NameRef('a')
    b = ir.NameRef('b')
    c = ir.NameRef('c')
    sub_a_b = ir.SUB(a, b)
    a_sub_a_b = ir.ADD(sub_a_b, c)
    expected_walk_order = (a, b, sub_a_b, c, a_sub_a_b)
    actual_walk_order = tuple(walk_expr(a_sub_a_b))
    assert expected_walk_order == actual_walk_order


def test_simple_walk_parameters():
    a = ir.NameRef('a')
    b = ir.NameRef('b')
    c = ir.NameRef('c')
    sub_a_b = ir.SUB(a, b)
    a_sub_a_b = ir.ADD(sub_a_b, c)
    expected_walk_order = (a, b, c)
    actual_walk_order = tuple(walk_parameters(a_sub_a_b))
    assert expected_walk_order == actual_walk_order
