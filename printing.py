from functools import singledispatchmethod

import ir


# Zip and Enumerate aren't really used explicitly elsewhere now
# so these are just sentinel classes

class Enumerate:
    def __init__(self, iterables, start):
        self.iterables = iterables
        self.start = start


class prettystringify:
    """
    The pretty printer is intended as a way to show the state of the IR in a way that resembles a
    typical source representation.

    
    """

    def __call__(self, node):
        return self.visit(node)

    def parenthesize_expr(self, node):
        if isinstance(node, ir.Expression):
            return f"({self.visit(node)})"
        else:
            return self.visit(node)

    @singledispatchmethod
    def visit(self, node):
        # handle non-parametric statements
        if isinstance(node, ir.EllipsisNode):
            return "..."
        return node.__class__.__name__.lower()

    @visit.register
    def _(self, node: Enumerate):
        if node.start != ir.IntNode(0):
            s = f"enumerate({self.visit(node.iterables)}, {self.visit(node.start)})"
        else:
            s = f"enumerate({self.visit(node.iterables)})"
        return s

    @visit.register
    def _(self, node: ir.AttributeRef):
        return self.visit(node.value)

    @visit.register
    def _(self, node: ir.NameRef):
        return node.name

    @visit.register
    def _(self, node: ir.ShapeRef):
        if node.dim is None:
            return self.visit(node.array)
        else:
            arr = self.visit(node.array)
            return f"{arr}.shape[{self.visit(node.dim)}]"

    @visit.register
    def _(self, node: ir.MinConstraint):
        args = ", ".join(self.visit(arg) for arg in node.constraints)
        return f"min({args})"

    @visit.register
    def _(self, node: ir.Assign):
        target = self.visit(node.target)
        value = self.visit(node.value)
        if node.in_place:
            return self.visit(node.value)
        else:
            return f"{target} = {value}"

    @visit.register
    def _(self, node: ir.IfExpr):
        s = f"{self.visit(node.if_expr)} if {self.visit(node.test)}"
        if node.else_expr is not None:
            s += f" {self.visit(node.else_expr)}"
        return s

    @visit.register
    def _(self, node: ir.BoolNode):
        return str(node.value)

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
        left = self.parenthesize_expr(node.left)
        right = self.parenthesize_expr(node.right)
        return f"{left} {node.op} {right}"

    @visit.register
    def _(self, node: ir.BoolOp):
        j = f" {node.op} "
        s = j.join(self.parenthesize_expr(operand) for operand in node.operands)
        if node.op == "and" and len(node.operands) > 1:
            if all(isinstance(operand, ir.BinOp) for operand in node.operands):
                if all(operand.op in ir.compareops for operand in node.operands):
                    if all(first.right == sec.left for (first, sec) in zip(node.operands[:-1], node.operands[1:])):
                        # We have a chained comparison, like a < b < c
                        s = f"{self.parenthesize_expr(node.operands[0].left)} " \
                            f"{node.operands[0].op} {self.parenthesize_expr(node.operands[0].right)}"
                        for expr in node.operands[1:]:
                            s += f" {expr.op} {self.parenthesize_expr(expr.right)}"
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
        if node.stop is None:
            # rarely used, just marking enumerate without full context
            s = "enumerate()"
        elif node.start == ir.IntNode(0):
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


def rebuild_enumerate_nesting(assigns):
    zipped_targets = []
    zipped_values = []
    for index, (target, value) in enumerate(assigns):
        value = value
        if isinstance(value, ir.Counter) and value.stop is None and value.step == ir.IntNode(1):
            # hit enumerate, wrap remainder
            t, v = rebuild_enumerate_nesting(assigns[index + 1:])
            zipped_values.append(Enumerate(v, value.start))
            zipped_targets.append(target)
            zipped_targets.append(t)
            break
        else:
            zipped_targets.append(target)
            zipped_values.append(value)
    if len(zipped_targets) == 1:
        zipped_targets = zipped_targets[0]
    else:
        zipped_targets = ir.Tuple(tuple(zipped_targets))
    if len(zipped_values) == 1:
        zipped_values = zipped_values[0]
    else:
        zipped_values = ir.Zip(tuple(zipped_values))
    return zipped_targets, zipped_values


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
    def _(self, node: list):
        for stmt in node:
            self.visit(stmt)

    @visit.register
    def _(self, node: ir.ForLoop):
        assigns = node.assigns
        if len(assigns) == 1:
            tar, it = assigns[0]
            tar = self.format(tar)
            it = self.format(it)
        else:
            # These are typically generated as flat assignments to whatever degree is possible,
            # but we still have to rebuild enumerate constructs
            tar, it = rebuild_enumerate_nesting(assigns)
            tar = self.format(tar)
            it = self.format(it)
        print(f"{self.leading}for {tar} in {it}:")
        self.indent()
        if node.body:
            self.visit(node.body)
        else:
            print(f"{self.leading}pass")
        self.unindent()

    @visit.register
    def _(self, node: ir.WhileLoop):
        test = self.format(node.test)
        print(f"{self.leading}while {test}:")
        self.indent()
        if node.body:
            self.visit(node.body)
        else:
            print(f"{self.leading}pass")
        self.unindent()

    @visit.register
    def _(self, node: ir.IfElse):
        test = self.format(node.test)
        print(f"{self.leading}if {test}:")
        if not node.if_branch:
            self.indent()
            print(f"{self.leading}pass")
            self.unindent()
        else:
            self.indent()
            self.visit(node.if_branch)
            self.unindent()
        if node.else_branch:
            possible_elif = node.else_branch[0]
            if isinstance(possible_elif, ir.IfElse) and len(node.else_branch) == 1:
                self.visit(possible_elif)
            else:
                print(f"{self.leading}else:")
                self.indent()
                self.visit(node.else_branch)
                self.unindent()

    @visit.register
    def _(self, node: ir.StmtBase):
        s = self.format(node)
        print(f"{self.leading}{s}")

    @visit.register
    def _(self, node: ir.SingleExpr):
        e = self.format(node.expr)
        print(f"{self.leading}{e}")
