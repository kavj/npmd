from functools import singledispatchmethod
from contextlib import contextmanager
from itertools import islice

import ir
import type_resolution as tr


#     https://docs.python.org/3/reference/expressions.html#operator-precedence

binop_ordering = {"**": 1, "*": 3, "@": 3, "/": 3, "//": 3, "%": 3, "+": 4, "-": 4, "<<": 5, ">>": 5, "&": 6,
                  "^": 7, "|": 8, "in": 9, "not in": 9, "<": 9, "<=": 9, ">": 9, ">=": 9, "!=": 9,
                  "==": 9}

# Todo: Given the boolean refactoring, not should probably derive from BoolOp, similar to TRUTH.

# Todo: Pretty printer should provide most of the infrastructure for C code gen. For plain C, most of the statement
#       structure used is the same, so this should be handled as much as posible via different expression visitors.
#       I'll also need to check for differences in operator precedence.

# Note, python docs don't specify truth precedence, but it should match logical "not"

scalar_pretty_types = {tr.Int32: "numpy.int32",
                       tr.Int64: "numpy.int64",
                       tr.Float32: "numpy.float32",
                       tr.Float64: "numpy.float64",
                       tr.Predicate32: "32_bit_mask",
                       tr.Predicate64: "64_bit_mask",
                       tr.BoolType: "bool"}


def get_pretty_scalar_type(t):
    scalar_type = scalar_pretty_types.get(t)
    return scalar_type


def get_pretty_type(t):
    if isinstance(t, ir.ArrayType):
        scalar_type = get_pretty_scalar_type(t.dtype)
        assert scalar_type is not None
        pt = f"numpy.ndarray[{scalar_type}]"
    else:
        pt = get_pretty_scalar_type(t)
    return pt


def parenthesized(formatted):
    return f"({formatted})"


class pretty_formatter:
    """
    The pretty printer is intended as a way to show the state of the IR in a way that resembles a
    typical source representation.

    Note: This will parenthesize some expressions that are unsupported yet accepted by plain Python.
          It's designed this way, because the alternative is more confusing.

    """

    def __call__(self, node):
        expr = self.visit(node)
        return expr

    @singledispatchmethod
    def visit(self, node):
        msg = f"No method to format node: {node}."
        raise NotImplementedError(msg)

    @visit.register
    def _(self, node: ir.Length):
        expr = self.visit(node.value)
        return f"len({expr})"

    @visit.register
    def _(self, node: ir.Max):
        args = ", ".join(self.visit(arg) for arg in node.values)
        return f"max({args})"

    @visit.register
    def _(self, node: ir.Min):
        args = ", ".join(self.visit(arg) for arg in node.values)
        return f"min({args})"

    @visit.register
    def _(self, node: ir.Ternary):
        test = self.visit(node.test)
        if isinstance(node.test, (ir.Ternary, ir.Tuple)):
            test = parenthesized(test)
        if_expr = self.visit(node.if_expr)
        if isinstance(node.if_expr, (ir.Ternary, ir.Tuple)):
            if_expr = parenthesized(if_expr)
        else_expr = self.visit(node.else_expr)
        if isinstance(node.else_expr, (ir.Ternary, ir.Tuple)):
            else_expr = parenthesized(else_expr)
        expr = f"{if_expr} if {test} else {else_expr}"
        return expr

    @visit.register
    def _(self, node: ir.BoolConst):
        return str(node.value)

    @visit.register
    def _(self, node: ir.IntConst):
        return str(node.value)

    @visit.register
    def _(self, node: ir.FloatConst):
        return str(node.value)

    @visit.register
    def _(self, node: ir.StringConst):
        return f"\"{node.value}\""

    @visit.register
    def _(self, node: ir.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op
        if not node.in_place:
            op_ordering = binop_ordering[op]
            if isinstance(node.left, ir.BinOp):
                if op_ordering < binop_ordering[node.left.op]:
                    left = parenthesized(left)
            elif isinstance(node.left, ir.UnaryOp):
                if op == "**":
                    left = parenthesized(left)
            elif isinstance(node.left, (ir.BoolOp, ir.CompareOp, ir.Ternary, ir.Tuple)):
                left = parenthesized(left)
            if isinstance(node.right, ir.BinOp):
                if op_ordering < binop_ordering[right.op]:
                    left = parenthesized(right)
            elif isinstance(node.right, ir.UnaryOp):
                if op == "**":
                    left = parenthesized(left)
            elif isinstance(node.right, (ir.BoolOp, ir.CompareOp, ir.Ternary, ir.Tuple)):
                right = parenthesized(right)
        expr = f"{left} {op} {right}"
        return expr

    @visit.register
    def _(self, node: ir.CompareOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.left, (ir.BoolOp, ir.CompareOp, ir.Ternary, ir.Tuple)):
            left = parenthesized(left)
        if isinstance(node.right, (ir.BoolOp, ir.CompareOp, ir.Ternary, ir.Tuple)):
            right = parenthesized(right)
        expr = f"{left} {node.op} {right}"
        return expr

    @visit.register
    def _(self, node: ir.AND):
        groups = []
        start = 0
        count = len(node.operands)
        assert count > 1
        while start < count:
            # group things like a < b < c
            # whether they arose from ir changes or input source
            first = node.operands[0]
            if isinstance(first, ir.CompareOp):
                cmp_op = first.op
                group = [first.left, first.right]
                prev_rhs = first.right
                for operand in islice(node.operands, start+1, None):
                    if not (isinstance(operand, ir.CompareOp) and cmp_op == operand.op and operand.left == prev_rhs):
                        break
                    group.append(operand.right)
                    prev_rhs = operand.right
                groups.append((group, cmp_op))
                start += len(group) - 1
            else:
                # something else, anded
                cmp_op = None
                groups.append((first, cmp_op))
        operands = []
        expr = None
        for group, cmp_op in groups:
            if cmp_op is not None:
                op = f" {cmp_op} "
                chain = op.join(self.visit(suboperand) for suboperand in group)
                operands.append(chain)
            else:
                # single expression
                assert isinstance(group, ir.ValueRef)
                formatted = self.visit(group)
                if isinstance(group, (ir.AND, ir.OR, ir.Ternary, ir.Tuple)):
                    formatted = parenthesized(formatted)
                operands.append(formatted)
            expr = "and ".join(operand for operand in operands)
        assert expr is not None
        return expr

    @visit.register
    def _(self, node: ir.OR):
        operands = []
        for operand in node.operands:
            formatted = self.visit(operand)
            if isinstance(operand, (ir.Ternary, ir.Tuple)):
                formatted = parenthesized(formatted)
            operands.append(formatted)
        expr = " or ".join(operand for operand in operands)
        return expr

    @visit.register
    def _(self, node: ir.NOT):
        formatted = self.visit(node.operand)
        if isinstance(node.operand, (ir.AND, ir.OR, ir.Ternary)):
            formatted = parenthesized(formatted)
        expr = f"not {formatted}"
        return expr

    @visit.register
    def _(self, node: ir.TRUTH):
        formatted = self.visit(node.operand)
        if node.constant:
            if not isinstance(node, ir.BoolConst):
                # We don't distinguish between bools and predicates here in
                # truth testing, since Python doesn't have any notion of
                # predicate types.
                formatted = f"bool({formatted})"
        return formatted

    @visit.register
    def _(self, node: ir.NameRef):
        expr = node.name
        return expr

    @visit.register
    def _(self, node: ir.Call):
        func_name = self.visit(node.func)
        args = ", ".join(self.visit(arg) for arg in node.args)
        func = f"{func_name}({args})"
        return func

    @visit.register
    def _(self, node: ir.Reversed):
        return f"reversed({self.visit(node.iterable)})"

    @visit.register
    def _(self, node: ir.Subscript):
        s = f"{self.visit(node.value)}[{self.visit(node.slice)}]"
        return s

    @visit.register
    def _(self, node: ir.AffineSeq):
        # Initial input source may not be easily disernible,
        # print as range
        start = self.visit(node.start)
        stop = self.visit(node.stop) if node.stop is not None else f"None"
        step = self.visit(node.step)
        return f"range({start}, {stop}, {step})"

    @visit.register
    def _(self, node: ir.Tuple):
        elements = []
        for e in node.elements:
            expr = self.visit(e)
            # parenthesize nested tuples, leave everything else
            if isinstance(e, ir.Tuple):
                expr = parenthesized(expr)
            elements.append(expr)
        s = ", ".join(e for e in elements)
        return s

    @visit.register
    def _(self, node: ir.UnaryOp):
        op = node.op
        operand = self.visit(node.operand)
        if isinstance(node.operand, ir.BinOp) and not node.operand.in_place:
            if node.operand.op != "**":
                operand = parenthesized(operand)
        elif isinstance(node.operand, (ir.UnaryOp, ir.BoolOp, ir.Ternary)):
            # if we have an unfolded double unary expression such as --,
            # '--expr' would be correct but it's visually jarring. Adding
            # unnecessary parentheses makes it '-(-expr)'.
            operand = parenthesized(operand)
        expr = f"{op}({operand})"
        return expr

    @visit.register
    def _(self, node: ir.Enumerate):
        iterable = self.visit(node.iterable)
        if node.start == ir.Zero:
            expr = f"enumerate({iterable})"
        else:
            start = self.visit(node.start)
            expr = f"enumerate({iterable}, {start})"
        return expr

    @visit.register
    def _(self, node: ir.Zip):
        exprs = []
        for elem in node.elements:
            formatted = self.visit(elem)
            if isinstance(elem, ir.Tuple):
                # This nesting is unsupported elsewhere, but this
                # would be a confusing place to throw an error.
                formatted = parenthesized(formatted)
            exprs.append(formatted)
        # handle case of enumerate
        expr = ", ".join(e for e in exprs)
        expr = f"zip({expr})"
        return expr


class pretty_printer:
    """
    Pretty prints tree. 
    Inserts pass on empty if statements or for/while loops.

    """

    def __init__(self, single_indent="    ", print_annotations=True):
        self.indent = ""
        self._increment = len(single_indent)
        self._single_indent = single_indent
        self.print_annotations = print_annotations
        self.format = pretty_formatter()
        self.symbols = None

    def __call__(self, tree, symbols):
        assert self.indent == ""
        with self.symbols_loaded(symbols):
            self.visit(tree)

    @contextmanager
    def symbols_loaded(self, symbols):
        assert self.symbols is None
        self.symbols = symbols
        yield
        self.symbols = None

    @contextmanager
    def indented(self):
        self.indent = f"{self.indent}{self._single_indent}"
        yield
        self.indent = self.indent[:-self._increment]

    def print_line(self, as_str):
        line = f"{self.indent}{as_str}"
        print(line)

    def make_elif(self, node: ir.IfElse):
        assert isinstance(node, ir.IfElse)
        test = self.format(node.test)
        if_expr = f"elif {test}:"
        self.print_line(if_expr)
        with self.indented():
            if node.if_branch:
                self.visit(node.if_branch)
            else:
                self.print_line("pass")
        if node.else_branch:
            # Make another elif if all conditions are met
            if len(node.else_branch) == 1:
                first, = node.else_branch
                if isinstance(first, ir.IfElse):
                    self.make_elif(first)
                    return
            self.print_line("else:")
            with self.indented():
                self.visit(node.else_branch)

    @singledispatchmethod
    def visit(self):
        raise NotImplementedError

    @visit.register
    def _(self, node: ir.ModImport):
        module = self.format(node.module)
        module_alias = self.format(node.as_name)
        if module == module_alias:
            as_str = f"import {module}"
        else:
            as_str = f"import {module} as {module_alias}"
        self.print_line(as_str)

    @visit.register
    def _(self, node: ir.NameImport):
        module = self.visit(node.module)
        imported_name = self.visit(node.name)
        import_alias = self.visit(node.as_name)
        if imported_name == import_alias:
            as_str = f"from {module} import {imported_name}"
        else:
            as_str = f"from {module} import {imported_name} as {import_alias}"
        self.print_line(as_str)

    @visit.register
    def _(self, node: ir.Return):
        if node.value is None:
            self.print_line("return")
        else:
            expr = self.format(node.value)
            stmt = f"return {expr}"
            self.print_line(stmt)

    @visit.register
    def _(self, node: ir.Module):
        for f in node.functions:
            print('\n')
            self.visit(f)
            print('\n')

    @visit.register
    def _(self, node: ir.Function):
        name = node.name
        args = ", ".join(self.format(arg) for arg in node.args)
        header = f"def {name}({args}):"
        self.print_line(header)
        with self.indented():
            self.visit(node.body)

    @visit.register
    def _(self, node: list):
        for stmt in node:
            self.visit(stmt)

    @visit.register
    def _(self, node: ir.ForLoop):
        target = self.format(node.target)
        iterable = self.format(node.iterable)
        stmt = f"for {target} in {iterable}:"
        self.print_line(stmt)
        with self.indented():
            if node.body:
                self.visit(node.body)
            else:
                # If all loop body statements are
                # dead/unreachable, append pass
                self.print_line("pass")

    @visit.register
    def _(self, node: ir.WhileLoop):
        test = self.format(node.test)
        stmt = f"while {test}:"
        self.print_line(stmt)
        with self.indented():
            self.visit(node.body)

    @visit.register
    def _(self, node: ir.IfElse):
        test = self.format(node.test)
        stmt = f"if {test}:"
        self.print_line(stmt)
        with self.indented():
            if node.if_branch:
                self.visit(node.if_branch)
            else:
                self.print_line("pass")
        if node.else_branch:
            # Make elif if all conditions are met
            if len(node.else_branch) == 1:
                first, = node.else_branch
                if isinstance(first, ir.IfElse):
                    self.make_elif(first)
                    return
            self.print_line("else:")
            with self.indented():
                self.visit(node.else_branch)

    @visit.register
    def _(self, node: ir.Assign):
        if node.in_place:
            stmt = self.format(node.value)
        else:
            target = node.target
            formatted_target = self.format(node.target)
            formatted_value = self.format(node.value)
            if self.print_annotations and isinstance(target, ir.NameRef):
                type_ = self.symbols.check_type(target)
                if type_ is not None:
                    # This is None if no typed symbol is registered
                    # This will be an error later.
                    pretty_type = get_pretty_type(type_)
                    if pretty_type is not None:
                        formatted_target = f"{formatted_target}: {pretty_type}"
            stmt = f"{formatted_target} = {formatted_value}"
        self.print_line(stmt)

    @visit.register
    def _(self, node: ir.Continue):
        self.print_line("continue")

    @visit.register
    def _(self, node: ir.Break):
        self.print_line("break")

    @visit.register
    def _(self, node: ir.SingleExpr):
        expr = self.format(node.expr)
        self.print_line(expr)
