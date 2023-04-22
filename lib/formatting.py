import numpy
import lib.ir as ir

from functools import singledispatchmethod
from lib.errors import CompilerError


#     https://docs.python.org/3/reference/expressions.html#operator-precedence

binop_ordering = {"**": 1, "*": 3, "@": 3, "/": 3, "//": 3, "%": 3, "+": 4, "-": 4, "<<": 5, ">>": 5, "&": 6,
                  "^": 7, "|": 8, "in": 9, "not in": 9, "<": 9, "<=": 9, ">": 9, ">=": 9, "!=": 9,
                  "==": 9}


compare_ops = {ir.EQ: "==",
               ir.NE: "!=",
               ir.LT: "<",
               ir.LE: "<=",
               ir.GT: ">",
               ir.GE: ">=",
               ir.IN: "in",
               ir.NOTIN: "not in"
               }

binop_ops = {ir.ADD: "+",
             ir.SUB: "-",
             ir.MULT: "*",
             ir.TRUEDIV: "/",
             ir.FLOORDIV: "//",
             ir.MOD: '%',
             ir.POW: "**",
             }


unary_ops = {
    ir.USUB: "-",
    ir.UINVERT: "~"
}


class PrettyFormatter:
    """
    The pretty printer is intended as a way to show the state of the IR in a way that resembles a
    typical source representation.

    Note: This will parenthesize some expressions that are unsupported yet accepted by plain Python.
          It's designed this way, because the alternative is more confusing.

    """

    def __call__(self, node, truncate_after=None):
        expr = self.visit(node)
        if truncate_after is not None and len(expr) > truncate_after:
            # Todo: check how this renders with parentheses
            expr = f'{expr[:truncate_after]}...'
        return expr

    def parenthesized(self, expr: ir.ValueRef):
        formatted = self.visit(expr)
        if isinstance(expr, ir.Expression):
            formatted = f'({formatted})'
        return formatted

    @singledispatchmethod
    def visit(self, node):
        msg = f"No method to format node: {node}."
        raise NotImplementedError(msg)

    @visit.register
    def _(self, node: ir.ArrayAlloc):
        exprs = ', '.join(self.visit(subexpr) for subexpr in node.subexprs)
        return f'numpy.empty({exprs})'

    @visit.register
    def _(self, node: ir.ArrayFill):
        return f'{self.visit(node.array)}[...] = {self.visit(node.fill_value)}'

    # adding formatting support for non-branching statements in order to print graph nodes
    @visit.register
    def _(self, node: ir.Assign):
        target = self.visit(node.target)
        value = self.visit(node.value)
        formatted = f'{target} = {value}'
        return formatted

    @visit.register
    def _(self, node: ir.Function):
        args = ', '.join(arg.name for arg in node.args)
        return f'def {node.name}({args})'

    @visit.register
    def _(self, node: ir.IfElse):
        test = self.visit(node.test)
        return f'if {test}'

    @visit.register
    def _(self, node: ir.ForLoop):
        target = self.visit(node.target)
        iterable = self.visit(node.iterable)
        return f'for {target} in {iterable}'

    @visit.register
    def _(self, node: ir.WhileLoop):
        test = self.visit(node.test)
        return f'while {test}'

    @visit.register
    def _(self, node: ir.InPlaceOp):
        left = node.target
        right = None
        for right in node.value.subexprs:
            # if there's a distinct one, pick it
            if right is not left:
                break
        assert right is not None
        op = binop_ops[type(node.value)]
        left = self.visit(left)
        right = self.visit(right)
        formatted = f'{left} {op}= {right}'
        return formatted

    @visit.register
    def _(self, node: ir.SingleExpr):
        return self.visit(node.value)

    @visit.register
    def _(self, node: ir.Break):
        return 'break'

    @visit.register
    def _(self, node: ir.Continue):
        return 'continue'

    @visit.register
    def _(self, node: ir.Return):
        value = self.visit(node.value)
        return f'return {value}'

    @visit.register
    def _(self, node: ir.NoneRef):
        return 'None'

    @visit.register
    def _(self, node: numpy.dtype):
        return str(node)

    @visit.register
    def _(self, node: ir.ArrayType):
        return str(node)

    @visit.register
    def _(self, node: ir.ArrayInitializer):
        if isinstance(node.fill_value, ir.NoneRef):
            name = 'empty'
        elif node.fill_value == ir.Zero:
            name = 'zeros'
        elif node.fill_value == ir.One:
            name = 'ones'
        else:
            msg = f'No name initializer will fill value "{self.visit(node.fill_value)}"'
            raise CompilerError(msg)
        if isinstance(node.shape, ir.TUPLE):
            shape = self.parenthesized(node.shape)
        else:
            shape = self.visit(node.shape)
        formatted = f'{name}({shape}, {node.dtype})'
        return formatted

    @visit.register
    def _(self, node: ir.SingleDimRef):
        expr = self.visit(node.base)
        if node.dim == ir.Zero:
            return f"len({expr})"
        else:
            dim = self.visit(node.dim)
            if isinstance(node.base, ir.NameRef):
                formatted = f"{expr}.shape[{dim}]"
            else:
                formatted = f'({expr}).shape[{dim}]'
            return formatted

    @visit.register
    def _(self, node: ir.MaxOf):
        args = ", ".join(self.visit(arg) for arg in node.subexprs)
        return f"max({args})"

    @visit.register
    def _(self, node: ir.MAX):
        args = ", ".join(self.visit(arg) for arg in node.subexprs)
        return f"max({args})"

    @visit.register
    def _(self, node: ir.MinOf):
        args = ", ".join(self.visit(arg) for arg in node.subexprs)
        return f"min({args})"

    @visit.register
    def _(self, node: ir.MIN):
        args = ", ".join(self.visit(arg) for arg in node.subexprs)
        return f"min({args})"

    @visit.register
    def _(self, node: ir.Slice):
        start = '' if isinstance(node.start, ir.NoneRef) else self.visit(node.start)
        stop = '' if isinstance(node.stop, ir.NoneRef) else self.visit(node.stop)
        step = '' if isinstance(node.step, ir.NoneRef) else self.visit(node.step)
        return f'Slice({start}:{stop}:{step})'

    @visit.register
    def _(self, node: ir.SELECT):
        (predicate, on_true, on_false) = (self.parenthesized(term)
                                          if isinstance(term, (ir.SELECT, ir.TUPLE)) else self.visit(term)
                                          for term in (node.predicate, node.on_true, node.on_false))
        expr = f"{on_true} if {predicate} else {on_false}"
        return expr

    @visit.register
    def _(self, node: ir.CAST):
        type_info = node.target_type
        output = f'{str(type_info)}({node.value})'
        return output

    @visit.register
    def _(self, node: ir.CONSTANT):
        return str(node.value)

    @visit.register
    def _(self, node: ir.StringConst):
        return f"\"{node.value}\""

    @visit.register
    def _(self, node: ir.BinOp):
        op = binop_ops[type(node)]
        op_ordering = binop_ordering[op]
        terms = []
        for term in node.subexprs:
            if isinstance(term, ir.BinOp):
                term_op = binop_ops[type(term)]
                if op_ordering < binop_ordering[term_op]:
                    term = self.parenthesized(term)
                else:
                    term = self.visit(term)
            elif isinstance(term, ir.UnaryOp):
                if op == "**":
                    term = self.parenthesized(term)
            elif isinstance(term, (ir.BoolOp, ir.CompareOp, ir.SELECT, ir.TUPLE)):
                term = self.parenthesized(term)
            else:
                term = self.visit(term)
            terms.append(term)
        left, right = terms
        expr = f"{left} {op} {right}"
        return expr

    @visit.register
    def _(self, node: ir.CompareOp):
        terms = []
        op = compare_ops[type(node)]
        for term in node.subexprs:
            if isinstance(term, (ir.BoolOp, ir.CompareOp, ir.SELECT, ir.TUPLE)):
                term = self.parenthesized(term)
            else:
                term = self.visit(term)
            terms.append(term)
        left, right = terms
        expr = f"{left} {op} {right}"
        return expr

    @visit.register
    def _(self, node: ir.AND):
        # Todo: move rebalancing to external pass
        expr = " and ".join(self.visit(operand) for operand in node.subexprs)
        return expr

    @visit.register
    def _(self, node: ir.OR):
        expr = " or ".join(self.visit(operand) for operand in node.subexprs)
        return expr

    @visit.register
    def _(self, node: ir.NOT):
        formatted = self.visit(node.operand)
        if isinstance(node.operand, (ir.AND, ir.OR, ir.SELECT)):
            formatted = self.parenthesized(formatted)
        expr = f"not {formatted}"
        return expr

    @visit.register
    def _(self, node: ir.TRUTH):
        formatted = self.visit(node.operand)
        if not isinstance(node, ir.CONSTANT):
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
        func_name = node.func
        args = ", ".join(self.visit(arg) for arg in node.args.subexprs)
        func = f"{func_name}({args})"
        return func

    @visit.register
    def _(self, node: ir.Reversed):
        return f"reversed({self.visit(node.iterable)})"

    @visit.register
    def _(self, node: ir.Subscript):
        if isinstance(node.index, ir.Subscript):
            index = ", ".join(self.visit(e) for e in node.index.subexprs)
        else:
            index = self.visit(node.index)
        base = self.visit(node.value)
        s = f"{base}[{index}]"
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
    def _(self, node: ir.TUPLE):
        elements = []
        for e in node.subexprs:
            # parenthesize nested tuples, leave everything else
            if isinstance(e, ir.TUPLE):
                expr = self.parenthesized(e)
            else:
                expr = self.visit(e)
            elements.append(expr)
        s = ", ".join(e for e in elements)
        return s

    @visit.register
    def _(self, node: ir.UnaryOp):
        op = unary_ops[type(node)]
        operand, = node.subexprs
        if isinstance(operand, ir.BinOp):
            if not isinstance(operand, ir.POW):
                operand = self.parenthesized(operand)
            else:
                operand = self.visit(operand)
        elif isinstance(operand, (ir.UnaryOp, ir.BoolOp, ir.SELECT)):
            # if we have an unfolded double unary expression such as --,
            # '--expr' would be correct but it's visually jarring. Adding
            # unnecessary parentheses makes it '-(-expr)'.
            operand = self.parenthesized(operand)
        else:
            operand = self.visit(operand)
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
            if isinstance(elem, ir.TUPLE):
                # This nesting is unsupported elsewhere, but this
                # would be a confusing place to throw an error.
                formatted = self.parenthesized(formatted)
            exprs.append(formatted)
        # handle case of enumerate
        expr = ", ".join(e for e in exprs)
        expr = f"zip({expr})"
        return expr
