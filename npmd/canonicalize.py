import itertools

import npmd.ir as ir

from npmd.errors import CompilerError
from npmd.pretty_printing import PrettyFormatter
from npmd.utils import extract_name


def replace_call(node: ir.Call):
    func_name = extract_name(node)
    if func_name == "numpy.ones":
        node = ir.ArrayInitializer(*node.args, ir.One)
    elif func_name == "numpy.zeros":
        node = ir.ArrayInitializer(*node.args, ir.Zero)
    elif func_name == "numpy.empty":
        node = ir.ArrayInitializer(*node.args, ir.NoneRef())
    elif func_name == "zip":
        node = ir.Zip(*node.args)
    elif func_name == "enumerate":
        node = ir.Enumerate(*node.args)
    elif func_name == "range":
        nargs = len(node.args)
        if nargs == 3:
            node = ir.AffineSeq(*node.args)
        elif nargs == 2:
            start, stop = node.args
            node = ir.AffineSeq(start, stop, ir.One)
        elif nargs == 1:
            stop, = node.args
            node = ir.AffineSeq(ir.Zero, stop, ir.One)
        else:
            pf = PrettyFormatter()
            msg = f"bad arg count for call to range {pf(node)}"
            raise CompilerError(msg)
    elif func_name == "len":
        assert len(node.args) == 1
        node = ir.SingleDimRef(node.args[0], ir.Zero)
    elif func_name == 'min':
        assert len(node.args) > 1
        terms = ir.MIN(node.args[0], node.args[1])
        for arg in itertools.islice(node.args, 2, None):
            terms = ir.MIN(terms, arg)
        node = terms
    elif func_name == 'max':
        assert len(node.args) > 1
        terms = ir.MAX(node.args[0], node.args[1])
        for arg in itertools.islice(node.args, 2, None):
            terms = ir.MAX(terms, arg)
        node = terms
    return node
