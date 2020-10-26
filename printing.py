from functools import singledispatchmethod

import ir


class prettystringify:
    """
    The pretty printer is intended as a way to show the state of the IR in a way that resembles a
    typical source representation.

    
    """

    def __call__(self, node):
        return self.visit(node)

    @singledispatchmethod
    def visit(self, node):
        # handle non-parametric statements
        if isinstance(node, ir.EllipsisNode):
            return "..."
        return node.__class__.__name__.lower()

    @visit.register
    def _(self, node: ir.Argument):
        s = node.name.name
        if node.annot is not None:
            s += f": {self.visit(node.annot)}"
        if node.defaultvalue is not None:
            s += f"={self.visit(node.defaultvalue)}"
        return s

    @visit.register
    def _(self, node: ir.AttributeRef):
        return self.visit(node.value)

    @visit.register
    def _(self, node: ir.NameRef):
        return node.name

    @visit.register
    def _(self, node: ir.Assign):
        target = self.visit(node.target)
        value = self.visit(node.value)
        if node.in_place:
            return self.visit(node.value)
        else:
            return f"{target} = {value}"

    @visit.register
    def _(self, node: ir.CascadeAssign):
        s = " = ".join(self.visit(t) for t in node.targets)
        v = self.visit(node.value)
        return f"{s} = {v}"

    @visit.register
    def _(self, node: ir.CompareOp):
        s = f"{self.visit(node.operands[0])}"
        for operand, op in zip(node.operands[1:], node.ops):
            s += f" {op} {self.visit(operand)}"
        return s

    @visit.register
    def _(self, node: ir.IfExpr):
        s = f"{self.visit(node.if_expr)} if {self.visit(node.test)}"
        if node.else_expr is not None:
            s += f" {self.visit(node.else_expr)}"
        return s

    @visit.register
    def _(self, node: ir.IntNode):
        return str(node.value)

    @visit.register
    def _(self, node: ir.FloatNode):
        return str(node.value)

    @visit.register
    def _(self, node: ir.StringNode):
        return f"\"{node.value}\""

    @visit.register
    def _(self, node: ir.NameRef):
        return node.name

    @visit.register
    def _(self, node: ir.BinOp):
        return f"{self.visit(node.left)} {node.op} {self.visit(node.right)}"

    @visit.register
    def _(self, node: ir.BoolOp):
        j = f" {node.op} "
        s = j.join(self.visit(operand) for operand in node.operands)
        return s

    @visit.register
    def _(self, node: ir.Call):
        funcname = node.funcname
        args = ", ".join(self.visit(arg) for arg in node.args)
        if args:
            return f"{funcname}({args})"
        else:
            return f"{funcname}()"

    @visit.register
    def _(self, node: ir.ModImport):
        if node.asname:
            return f"import {node.mod} as {node.asname}"
        else:
            return f"import {node.mod}"

    @visit.register
    def _(self, node: ir.NameImport):
        if node.asname:
            return f"from {node.mod} import {node.name}"
        else:
            return f"from {node.mod} import {node.name} as {node.asname}"

    @visit.register
    def _(self, node: ir.Return):
        if node.value is not None:
            return f"return {self.visit(node.value)}"
        else:
            return "return"

    @visit.register
    def _(self, node: ir.Reversed):
        return f"reversed({self.visit(node.iterable)})"

    @visit.register
    def _(self, node: ir.Subscript):
        s = self.visit(node.value)
        s += f"[{self.visit(node.slice)}]"
        return s

    @visit.register
    def _(self, node: ir.Counter):
        assert (node.stop is not None)
        if node.start == ir.IntNode(0):
            if node.step == ir.IntNode(1):
                s = f"range({self.visit(node.stop)})"
            else:
                s = f"range(0, {self.visit(node.stop)}, {self.visit(node.step)})"
        elif node.step == ir.IntNode(1):
            s = f"range({self.visit(node.start)}, {self.visit(node.stop)})"
        else:
            s = f"range({self.visit(node.start)}, {self.visit(node.stop)}, {self.visit(node.step)})"
        return s

    @visit.register
    def _(self, node: ir.Tuple):
        if len(node.elements) == 1:
            s = f"{self.visit(node.elements[0])},"
        else:
            s = f", ".join(self.visit(e) if not isinstance(e, ir.Tuple) else f"({self.visit(e)})"
                           for e in node.elements)
        return s

    @visit.register
    def _(self, node: ir.UnaryOp):
        if node.op == "not":
            return f"not {self.visit(node.operand)}"
        else:
            return f"{node.op}{self.visit(node.operand)}"

    @visit.register
    def _(self, node: ir.Zip):
        counter = node.elements[0]
        if not isinstance(counter, ir.Counter) or len(node.elements) != 2:
            # general zip
            s = ", ".join(self.visit(e) for e in node.elements)
            s = f"zip({s})"
        elif counter.step != ir.IntNode(1) or counter.stop is not None:
            # zip(range(...), iterable) as opposed to enumerate(iterable)
            s = f"zip({self.visit(counter)}, {self.visit(node.elements[1])})"
        elif counter.start == ir.IntNode(0):
            s = f"enumerate({self.visit(node.elements[1])})"
        else:
            s = f"enumerate({self.visit(node.elements[1])}, {self.visit(node.elements[0])})"
        return s

    @visit.register
    def _(self, node: ir.SingleExpr):
        return self.visit(node.expr)


class printtree:
    """
    Pretty prints tree. 
    Inserts pass on empty if statements or for/while loops.

    """

    def __call__(self, tree):
        self.leading = ""
        self.visit(tree)

    def __init__(self, spaces=4):
        self.leading = ""
        self.format = prettystringify()
        self.spaces = spaces
        self.pad = "".join(" " for _ in range(spaces))

    def indent(self):
        self.leading += self.pad

    def unindent(self):
        self.leading = self.leading[:-self.spaces]

    def visit_elif(self, node):
        test = self.format(node.test)
        print(f"{self.leading}elif {test}:")
        if node.empty_if:
            self.indent()
            print(f"{self.leading}pass")
            self.unindent()
        else:
            self.indent()
            for stmt in node.if_branch:
                self.visit(stmt)
            self.unindent()
        if len(node.else_branch) == 1 and isinstance(node.else_branch[0], ir.IfElse):
            self.visit_elif(node.else_branch[0])
        elif node.else_branch:
            print(f"{self.leading}else:")
            self.indent()
            for stmt in node.else_branch:
                self.visit(stmt)
            self.unindent()

    @singledispatchmethod
    def visit(self, node):
        raise NotImplementedError(f"No method to print node of type {type(node)}")

    @visit.register
    def _(self, node: ir.Module):
        for stmt in node.imports:
            self.visit(stmt)
        print('\n')
        for f in node.funcs:
            print('\n')
            self.visit(f)
            print('\n')

    @visit.register
    def _(self, node: ir.Function):
        name = node.name
        args = ", ".join(self.format(arg) for arg in node.args)
        print(f"{self.leading}def {name}({args}):")
        self.indent()
        for stmt in node.body:
            self.visit(stmt)
        self.unindent()

    @visit.register
    def _(self, node: ir.ForLoop):
        tar = self.format(node.target)
        it = self.format(node.iterable)
        print(f"{self.leading}for {tar} in {it}:")
        self.indent()
        if node.body:
            for stmt in node.body:
                self.visit(stmt)
        else:
            print(f"{self.leading}pass")
        self.unindent()

    @visit.register
    def _(self, node: ir.WhileLoop):
        test = self.format(node.test)
        print(f"{self.leading}while {test}:")
        self.indent()
        if node.body:
            for stmt in node.body:
                self.visit(stmt)
        else:
            print(f"{self.leading}pass")
        self.unindent()

    @visit.register
    def _(self, node: ir.IfElse):
        test = self.format(node.test)
        print(f"{self.leading}if {test}:")
        if node.empty_if:
            self.indent()
            print(f"{self.leading}pass")
            self.unindent()
        else:
            self.indent()
            for stmt in node.if_branch:
                self.visit(stmt)
            self.unindent()
        if node.else_branch:
            possible_elif = node.else_branch[0]
            if isinstance(possible_elif, ir.IfElse) and len(node.else_branch) == 1:
                self.visit(possible_elif)
            else:
                print(f"{self.leading}else:")
                self.indent()
                for stmt in node.else_branch:
                    self.visit(stmt)
                self.unindent()

    @visit.register
    def _(self, node: ir.StmtBase):
        s = self.format(node)
        print(f"{self.leading}{s}")

    @visit.register
    def _(self, node: ir.SingleExpr):
        e = self.format(node.expr)
        print(f"{self.leading}{e}")
