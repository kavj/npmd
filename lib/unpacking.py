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


def unpack_iter(target: ir.ValueRef, iterable: ir.ValueRef):
    if isinstance(target, ir.NameRef) and isinstance(iterable, (ir.NameRef, ir.AffineSeq, ir.Subscript)):
        yield target, iterable
    else:
        queued = [unpack_loop_one_level(target, iterable)]
        while queued:
            for sub_target, sub_iterable in queued[-1]:
                if isinstance(sub_target, ir.NameRef) and isinstance(sub_iterable, (ir.NameRef, ir.AffineSeq, ir.Subscript)):
                    yield sub_target, sub_iterable
                else:
                    queued.append(unpack_loop_one_level(sub_target, sub_iterable))
                    break
            else:
                # exhausted
                queued.pop()


def unpack_loop_iter(node: ir.ForLoop):
    yield from unpack_iter(node.target, node.iterable)


def unpack_assignment(target, value, pos):
    if isinstance(target, ir.TUPLE) and isinstance(value, ir.TUPLE):
        if target.length != value.length:
            msg = f"Cannot unpack {value} with {value.length} elements using {target} with {target.length} elements: " \
                  f"line {pos.line_begin}."
            raise ValueError(msg)
        for t, v in zip(target.subexprs, value.subexprs):
            yield from unpack_assignment(t, v, pos)
    else:
        yield target, value
