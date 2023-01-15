import lib.ir as ir

from lib.errors import CompilerError
from lib.formatting import PrettyFormatter


def get_unpacking_count(expr: ir.ValueRef):
    if isinstance(expr, (ir.TUPLE, ir.Zip)):
        return len(expr.elements)
    elif isinstance(expr, ir.Enumerate):
        return 2
    else:
        assert isinstance(expr, ir.ValueRef)
        return 1


def unpack_loop_one_level(target: ir.ValueRef, iterable: ir.ValueRef):

    if not isinstance(target, ir.TUPLE):
        pf = PrettyFormatter()
        pretty_target = pf(target)
        pretty_iterable = pf(iterable)
        msg = f'Only tuple targets are supported targets for loop iterator unpacking. Received target:' \
              f' "{pretty_target}" and iterable: "{pretty_iterable}"'
        raise CompilerError(msg)

    if not isinstance(iterable, (ir.Zip, ir.Enumerate)):
        pf = PrettyFormatter()
        pretty_iterable = pf(target)
        msg = f'Iterable: {pretty_iterable} is not unpackable.'
        raise CompilerError(msg)

    target_count = get_unpacking_count(target)
    iterable_count = get_unpacking_count(iterable)

    if target_count != iterable_count:
        pf = PrettyFormatter()
        pretty_target = pf(target)
        pretty_iterable = pf(iterable)
        msg = f'Cannot unpack iterable {pretty_iterable} with {iterable_count} elements into target {pretty_target} ' \
              f'with {target_count} elements.'
        raise CompilerError(msg)

    if isinstance(iterable, ir.Enumerate):
        return zip(target.elements, (ir.AffineSeq(iterable.start, None, ir.One), iterable.iterable))

    return zip(target.elements, iterable.elements)


def unpack_loop_iter(node: ir.ForLoop):
    if isinstance(node.target, ir.NameRef) and isinstance(node.iterable, (ir.NameRef, ir.AffineSeq)):
        yield node.target, node.iterable
    else:
        queued = [unpack_loop_one_level(node.target, node.iterable)]
        while queued:
            for target, iterable in queued[-1]:
                if isinstance(target, ir.NameRef) and isinstance(iterable, (ir.NameRef, ir.AffineSeq)):
                    yield target, iterable
                else:
                    queued.append(unpack_loop_one_level(target, iterable))
                    break
            else:
                # exhausted
                queued.pop()
