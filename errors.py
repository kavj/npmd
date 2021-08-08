from contextlib import contextmanager


# Skeleton

# Should be something like this. This should by default not print the entire compiler stack.


class CompilerError(Exception):
    """
    An exception type to specify errors, which are presumed to be semantic errors in input source
    unless explicitly labeled as internal errors. These are used to halt execution with an appropriate error
    message, without printing a stack trace that refers to compiler internals.
    """
    def __init__(self, *args):
        super().__init__(args)


@contextmanager
def module_context(print_stack=False):
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
