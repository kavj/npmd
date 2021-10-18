import ir
from functools import singledispatchmethod
from errors import CompilerError
from visitor import StmtVisitor


class ArrayAssignCheck(StmtVisitor):
    """
    Stub to check that arrays are not arbitrarily assigned outside of array creation routines.
    This simplifies bookkeeping and helps with nogil emulation of arrays
    Initialization checks are handled by reaching_check.
    """

    def __init__(self, ctx):
        self.ctx = ctx

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Assign):
        # If the left
        # determine if the left hand side refers to an array or sub-array
        # determine if the right hand side is an array creation routine.
        # check uniqueness of assignment, that is no array name contains more than one
        # out of place assignment
        pass

