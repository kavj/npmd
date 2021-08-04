from contextlib import contextmanager
from inspect import getsourcelines


# Skeleton

# Should be something like this. This should by default not print the entire compiler stack.


class CompilerError(Exception):
    """
    An exception that specifies an error in input source code. This is used to halt execution and generate
    error logging without printing out a stack trace of unrelated compiler internals.

    """
    def __init__(self, *args):
        super().__init__(args)


@contextmanager
def compiler_pass_context(entry, src, print_stack=False):
    try:
        yield
    except CompilerError as ce:
        print(ce)
        return


@contextmanager
def source_error_tracking(pos):
    try:
        yield
    except CompilerError as ce:
        msg = f"Line {pos}:"
        raise CompilerError(msg, str(ce))
