import ast

from contextlib import contextmanager
from functools import singledispatchmethod
from pathlib import Path
from typing import Optional

import ir

from errors import CompilerError
from symbol_table import func_symbol_table, module_symbol_table
from utils import extract_name


class CompilerContext:

    def __init__(self):
        self.current_module = None
        self.current_function = None
        self._modules = {}

    @contextmanager
    def function_scope(self, name):
        assert self.current_function is None
        # Check that current module declares this function
        func = self.current_module.lookup_func(name)
        if func is None:
            raise CompilerError(f"No entry for function {entry_point} in module {self.current_module.name}")
        self.current_function = func
        yield
        self.current_function = None

    @contextmanager
    def module_scope(self, name: ir.Module):
        assert self.current_module is None
        # create a module entry if we don't have one
        module = self._modules.get(name)
        if module is None:
            msg = f"No lookup for module {name}."
            raise RuntimeError(msg)
        self.current_module = module
        yield
        self.current_module = None

    def register_module_scope(self, symbol_table: module_symbol_table):
        name = symbol_table.name
        if name in self._modules:
            msg = f"Module name {name} shadows an existing module. This is unsupported."
            raise CompilerError(msg)
        self._modules[name] = symbol_table

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

    def register_unique_name(self, prefix):
        table = self.current_function
        prefix_ = extract_name(prefix)
        name = table.make_unique_name(prefix_)
        return name

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
