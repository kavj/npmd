import itertools

import ir

from visitor import walk_all, walk_assigns


def extract_name(name):
    return name.name if isinstance(name, ir.NameRef) else name


class symbol_gen:
    def __init__(self, existing):
        self.names = existing
        self.added = set()
        self.gen = itertools.count()

    def __contains__(self, item):
        if isinstance(item, ir.NameRef):
            item = item.name
        return item in self.names

    def make_unique_name(self, prefix):
        name = f"{prefix}_{next(self.gen)}"
        while name in self.names:
            name = f"{prefix}_{next(self.gen)}"
        self.names.add(name)
        return name


def build_symbols(entry):
    # grabs all names that can be declared at outmermost scope
    names = set()
    if isinstance(entry, ir.Function):
        for arg in entry.args:
            names.add(extract_name(arg))
    for stmt in walk_all(entry):
        if isinstance(stmt, ir.Assign):
            if isinstance(stmt.target, ir.NameRef):
                names.add(extract_name(stmt.target))
        elif isinstance(stmt, ir.ForLoop):
            for target, _ in walk_assigns(stmt):
                names.add(extract_name(target))
    return names
