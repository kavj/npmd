import builtins
import keyword

from functools import singledispatchmethod

import ir

from visitor import VisitorBase

keywords = frozenset(set(keyword.kwlist))
builtin_names = frozenset(set(dir(builtins)))


def shadows_builtin_name(name):
    return name in keywords or name in builtin_names


class TreeValidator(VisitorBase):
    """
    A visitor to perform early checks for common errors

    """
    reserved = frozenset(set(keyword.kwlist).union(set(dir(builtins))))

    def __call__(self, entry):
        self.entry = entry
        self.errors = []
        self.visit(entry)
        errors = self.errors
        self.entry = self.errors = None
        return errors

    @singledispatchmethod
    def visit(self, node):
        super().visit(node)

    @visit.register
    def _(self, node: ir.Function):
        if node.name in self.reserved:
            self.errors.append(f"Function: {node.name} shadows a keyword or builtin function")
        if node is self.entry:
            seen = set()
            for arg in node.args:
                if arg in seen:
                    self.errors.append(f"Duplciate argument {arg} in function {node.name}")
                elif arg in self.reserved:
                    self.errors.append(f"Argument {arg} in function {node.name} shadows a keyword or builtin function")
                seen.add(arg)
                self.visit(node.body)
        else:
            self.errors.append(f"Nested functions are not supported. Encountered {node.name}")

    @visit.register
    def _(self, node: ir.ForLoop):
        targets = set()
        iterables = set()
        for target, iterable in node.walk_assignments():
            if target in targets:
                # it's fine for iterables to appear in more than one location, particularly if tracking different
                # positions
                self.errors.append(f"duplicate target variable in for loop on line {node.pos.line_begin}")
            iterables.add(iterable)
        conflicts = targets.intersection(iterables)
        if conflicts:
            for conflict in conflicts:
                self.errors.append(f"{conflict} appears as both an iterable and target in for loop on line "
                                   f"{node.pos.line_begin}.")
        self.visit(node.body)

    @visit.register
    def _(self, node: ir.Assign):
        if isinstance(node.target, ir.NameRef):
            name = node.target.name
            if name in self.reserved:
                self.errors.append(f"Assignment target {name} at line {node.pos.line_begin} shadows a keyword or "
                                   f"builtin function")
