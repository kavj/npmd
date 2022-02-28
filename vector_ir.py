from dataclasses import dataclass
from typing import Optional, Tuple

import ir


@dataclass(frozen=True)
class load(ir.Expression):
    array: ir.ArrayRef
    index: ir.ValueRef
    predicate: Optional[ir.ValueRef] = None

    @property
    def subexprs(self):
        yield self.array
        yield self.index
        if self.predicate is not None:
            yield self.predicate


@dataclass(frozen=True)
class store(ir.Expression):
    array: ir.ArrayRef
    index: ir.ValueRef
    value: ir.ValueRef
    predicate: Optional[ir.ValueRef] = None
    
    @property
    def subexprs(self):
        yield self.array
        yield self.index
        yield self.value
        if self.predicate is not None:
            yield self.predicate


@dataclass(frozen=True)
class transpose(ir.Expression):
    rows: Tuple[ir.ValueRef,...]


@dataclass(frozen=True)
class blend(ir.Expression):
    base: ir.ValueRef
    alternate: ir.ValueRef
    predicate: ir.ValueRef
