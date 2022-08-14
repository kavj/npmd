import sys
import pytest

import numpy as np

import npmd.ir as ir

from npmd.errors import CompilerError
from npmd.folding import fold_constants, simplify_arithmetic


a = ir.wrap_constant(45)
b = ir.wrap_constant(45.0)
c = ir.wrap_constant(2**63-1)
some_str = ir.wrap_constant('blah')
float_min = ir.wrap_constant(sys.float_info.min)
float_max = ir.wrap_constant(sys.float_info.max)


def test_float_overflow():
    overflow = ir.MULT(float_max, ir.Two)
    res = fold_constants(overflow)
    inf = ir.wrap_constant(np.inf)
    assert res.matches(inf)
    assert res == inf


def test_integer_overflow():
    with pytest.raises(CompilerError):
        ir.wrap_constant(2**64)


def test_different_types_not_merged():
    pass


def test_constant_folding():
    pass


def test_folding_bool():
    pass


def test_cast_node_added():
    pass
