import lib.ir as ir
from lib.unpacking import unpack_loop_iter


#  unpacking is used all over the place, since the unrefined version is typically easier to analyze

def test_basic_loop():
    """
    test

    for v in a:
       pass

    :return:
    """
    a = ir.NameRef('a')
    v = ir.NameRef('v')
    header = ir.ForLoop(target=v, iterable=a, pos=ir.Position(0, 0, 0, 0))
    unpacked = tuple(unpack_loop_iter(header))
    assert len(unpacked) == 1
    pair, = unpacked
    assert len(pair) == 2
    assert pair == (v, a)


def test_enumerate_name():
    """
    try to unpack

    for i, v in enumerate(a):
        pass

    :return:
    """
    i = ir.NameRef('i')
    v = ir.NameRef('v')
    a = ir.NameRef('a')
    enumerated = ir.Enumerate(a)
    target = ir.TUPLE(i, v)
    affine = ir.AffineSeq(start=ir.Zero, stop=None, step=ir.One)
    header = ir.ForLoop(target=target, iterable=enumerated, pos=ir.Position(0, 0, 0, 0))
    unpacked = tuple(unpack_loop_iter(header))
    assert len(unpacked) == 2
    index, uiterable = unpacked
    assert len(index) == len(uiterable) == 2
    assert index == (i, affine)
    assert uiterable == (v, a)


def test_zip_names():
    """
    Tests left to right unpacking with zip
    :return:
    """
    u = ir.NameRef('u')
    v = ir.NameRef('v')
    w = ir.NameRef('w')
    a = ir.NameRef('a')
    b = ir.NameRef('b')
    c = ir.NameRef('c')
    zipped = ir.Zip(a, b, c)
    target = ir.TUPLE(u, v, w)
    header = ir.ForLoop(target=target, iterable=zipped, pos=ir.Position(0, 0, 0, 0))
    unpacking = tuple(unpack_loop_iter(header))
    assert unpacking == ((u, a), (v, b), (w, c))
