import ir

from symbol_table import symbol_table
from utils import extract_name

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
        yield
        self.current_module = None

    def check_type(self, name):
        name = extract_name(name)
        return self.current_function.types.get(name)

    def bind_type(self, name, type_):
        assert type_ is not None
        existing_type = self.check_type(name)
        if existing_type is not None:
            if existing_type != type_:
                msg = f"Conflicting types {existing_type} and {type_} for variable name {name}."
                raise CompilerError(msg)
        else:
            self.current_function.types[name] = type_

    def is_array(self, name):
        type_ = self.check_type(name)
        return isinstance(type_, ir.ArrayType)

    def is_typed(self, name):
        name = extract_name(name)
        return name in self.current_function.types

    @property
    def return_type(self):
        # Todo: stub
        return None
