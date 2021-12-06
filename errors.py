from contextlib import contextmanager


class CompilerError(Exception):
    """
    An exception type to specify errors, which are presumed to be semantic errors in input source
    unless explicitly labeled as internal errors. These are used to halt execution with an appropriate error
    message, without printing a stack trace that refers to compiler internals.
    """
    def __init__(self, *args):
        super().__init__(args)


@contextmanager
def error_context():
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
