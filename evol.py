from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Optional

import ir

from folding import fold_constants, fold_op, simplify_select
from symbol_table import SymbolTable
from type_checks import TypeHelper


@dataclass(frozen=True)
class SCEV:
    """
    scev with optional boundary conditions
    """
    start: ir.ValueRef
    step: ir.ValueRef
    stop: Optional[ir.ValueRef]
    array: Optional[ir.ValueRef]

    @property
    def trip_count_expression(self):
        if self.stop is not None and self.array is not None:
            ub = ir.MIN(self.stop, ir.Length(self.array))
            ub = fold_constants(ub)
            diff = fold_op(ir.SUB(ub, self.start))
        elif self.stop is not None:
            diff = fold_op(ir.SUB(self.stop, self.start))
        elif self.array is not None:
            diff = fold_op(ir.SUB(ir.Length(self.array), self.start))
        else:
            return None
        base = fold_op(ir.FLOORDIV(diff, self.step))
        fringe = fold_op(ir.MOD(diff, self.step))
        expr = simplify_select(ir.SELECT(fringe, base, fold_constants(ir.ADD(base, ir.One))))
        return expr


class AccessFuncCollect:

    def __init__(self, symbols: SymbolTable):
        self.typer = TypeHelper(symbols)

    @singledispatchmethod
    def extract_scev(self, node):
        raise NotImplementedError

    @extract_scev.register
    def _(self, node: ir.NameRef) -> Optional[SCEV]:
        t = self.typer.check_type(node)
        if isinstance(t, ir.ArrayType):
            return SCEV(start=ir.Zero, step=ir.One, stop=ir.SingleDimRef(node, ir.Zero), array=node)

    @extract_scev.register
    def _(self, node: ir.Subscript) -> Optional[SCEV]:
        index = node.index
        if isinstance(index, ir.Slice):
            stop = ir.MIN(index.stop, ir.SingleDimRef(node.value, ir.Zero))
            return SCEV(start=index.start, step=index.step, stop=stop, array=node.value)
        else:
            t = self.typer.check_type(node)
            if isinstance(t, ir.ArrayType):
                stop = ir.SingleDimRef(node.value, ir.One)
                return SCEV(start=index, step=ir.One, stop=stop, array=node.value)

    @extract_scev.register
    def _(self, node: ir.AffineSeq) -> SCEV:
        return SCEV(node.start, node.step, node.stop, None)
