import numpy as np

from dataclasses import dataclass
from functools import singledispatch
from typing import Dict, Union

import ir

from symbol_table import SymbolTable
from traversal import Exp
from type_checks import TypeHelper


@dataclass(frozen=True)
class Vector:
    dtype: np.dtype

# implement some datatypes for stuff that works differently

# Todo: vectorize "in" and "not in"
# Vector expressions always use 3 address form. This helps ensure sane implementation.


@dataclass(frozen=True)
class Broadcast(ir.Expression):
    """
    broadcast of some scalar. This
    """
    value: Union[ir.NameRef, ir.CONSTANT]


@dataclass(frozen=True)
class ArrayReduction(ir.Expression):
    value: Union[ir.NameRef, ir.BinOp, ir.CompareOp]
    op: str

    def __post_init__(self):
        assert isinstance(self.value, (ir.NameRef, ir.BinOp, ir.CompareOp))

    @property
    def subexprs(self):
        yield self.value

    def reconstruct(self, value):
        return super().reconstruct(value, self.op)


@dataclass(frozen=True)
class ArrayMin(ir.Expression):
    value: ir.NameRef


@dataclass(frozen=True)
class ArrayMax(ir.Expression):
    value: ir.NameRef


@dataclass(frozen=True)
class ArraySum(ir.Expression):
    value: Union[ir.NameRef, ir.Expression]


@dataclass(frozen=True)
class ArrayAnd(ir.Expression):
    value: Union[ir.NameRef, ir.Expression]


@dataclass(frozen=True)
class ArrayOr(ir.Expression):
    value: Union[ir.NameRef, ir.Expression]


# Todo: need vector arithmetic extensions..
#       also need broadcast


def render_vector_stmt(node: ir.Assign, symbols: SymbolTable):
    """
    this doesn't do nested expressions..
    :param node:
    :param symbols:
    :return:
    """

    # verify not nested
    if not isinstance(node, Broadcast) and not all(isinstance(subexpr, ir.NameRef) for subexpr in node.subexprs):
        msg = f"Vector Expressions must have 3 address form. This is probably a bug."
        raise TypeError(msg)


prefix = 'npyv_'

dtype_tag = {np.dtype('int8'): '_s8',
             np.dtype('int16'): '_s16',
             np.dtype('int32'): '_s32',
             np.dtype('int64'): '_s64',
             np.dtype('uint8'): '_u8',
             np.dtype('uint16'): '_u16',
             np.dtype('uint32'): '_u32',
             np.dtype('uint64'): '_u64',
             np.dtype('float32'): '_f32',
             np.dtype('float64'): '_f64'}
