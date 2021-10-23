import ir

from symbol_table import symbol_table

from contextlib import contextmanager
from functools import singledispatchmethod

class compiler_context:

    def __init__(self):
        self.current_module = None
        self.current_function = None
        self.symbols = {}

    @contextmanager
    def function_scope(self, entry_point: ir.Function):
        assert self.current_function is None
        # Check that current module declares this function
        self.current_function = entry_point
        yield
        self.current_function = None

    @contextmanager
    def module_scope(self, entry_point: ir.Module):
        assert self.current_module is None
        # create a module entry if we don't have one
        self.current_module = entry_point

    def check_type(self):
        # Todo: stub
        pass

    @property
    def return_type(self):
        # Todo: stub
        return None
