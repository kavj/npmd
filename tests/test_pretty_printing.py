
import npmd.ir as ir

from npmd.analysis import compute_element_count, find_array_length_expression
from npmd.pretty_printing import PrettyFormatter

formatter = PrettyFormatter()

a = ir.NameRef('a')
b = ir.NameRef('b')
c = ir.NameRef('c')
i = ir.NameRef('i')
j = ir.NameRef('j')
k = ir.NameRef('k')
m = ir.NameRef('m')
n = ir.NameRef('n')


def test_formatting():
    # multiply add with no nested
    basic_mul_add = ir.ADD(ir.MULT(a, b), c)
    formatted = formatter(basic_mul_add)
    assert '(' not in formatted
    assert ')' not in formatted
    # Now check if we include a nested add,
    # it inserts parentheses to avoid the multiplication taking precedence
    apb = ir.ADD(a, b)
    with_nested_add = ir.ADD(ir.MULT(apb, c), b)
    formatted = formatter(with_nested_add)
    subexpr = formatter(apb)
    assert subexpr in formatted
    # manually add parentheses and check it's s
    subexpr = f'({subexpr})'
    assert subexpr in formatted


def test_complex_len_expr():
    t0 = ir.Subscript(a, ir.Slice(j, k, m))
    count = find_array_length_expression(t0)
    formatted = formatter(count)
    # Ensure that the requisite components show up here
    # Note, this isn't the safest. Should be sufficient in real world cases
    # to enable wrapping and use 64 bit types

