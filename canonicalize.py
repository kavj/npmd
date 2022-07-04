import ir

from analysis import check_all_paths_return
from errors import CompilerError
from pretty_printing import PrettyFormatter
from symbol_table import SymbolTable
from type_checks import check_return_type
from utils import extract_name


def replace_call(node: ir.Call):
    func_name = extract_name(node)
    if func_name == "numpy.ones":
        node = ir.ArrayInitializer(*node.args, ir.One)
    elif func_name == "numpy.zeros":
        node = ir.ArrayInitializer(*node.args, ir.Zero)
    elif func_name == "numpy.empty":
        node = ir.ArrayInitializer(*node.args, ir.NoneRef)
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
        node = ir.Length(*node.args)
    return node


def patch_return(node: ir.Function, symbols: SymbolTable):
    always_terminated = check_all_paths_return(node.body)
    if not always_terminated:
        return_type = check_return_type(node, symbols)
        if isinstance(return_type, ir.NoneRef):
            if node.body:
                last_pos = node.body[-1].pos
                pos = ir.Position(last_pos.line_end + 1, last_pos.line_end + 2, col_begin=1, col_end=74)
            else:
                # Todo: not quite right but will revisit
                pos = ir.Position(1, 1, 1, 74)
            body = node.body.copy()
            body.append(ir.Return(value=ir.NoneRef(), pos=pos))
            repl = ir.Function(name=node.name, args=node.args, body=body)
            return repl
        else:
            # If return type is anything other than None and we don't explicitly return along
            # all paths, then this is unsafe
            msg = f"Function {node.name} does not return a value along all paths."
            raise CompilerError(msg)
    return node
