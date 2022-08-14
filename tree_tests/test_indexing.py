
def test_index_0(a):
    """
    enumerate usable index from dead variable
    :param a:
    :return:
    """

    for i, u in enumerate(a):
        print(u)


def test_index_1(a, n):
    """
    usable range
    :param a:
    :param n:
    :return:
    """

    for i, u in zip(range(n), a):
        print(i, u)


def test_index_2(a):
    """
    clobbered iterable
    :param a:
    :return:
    """

    for i, u in enumerate(a):
        i += 10
        print(i, u)


def test_index_3(a):
    """
    constant stride
    :param a:
    :return:
    """

    for u in a[::4]:
        print(u)


def test_index_4(a, i):
    """
    unknown stride
    :param a:
    :return:
    """

    for u in a[::i]:
        print(u)


def test_index_5(a, b, k):
    """
    unknown stride, mixed with similar range params
    This should detect that i is a suitable index and that it bounds the other 2

    """
    for t, u, v in zip(range(0, min(len(a), len(b)), k), a[::k], b[::k]):
        j = t * u
        print(j, u + v)


def test_index_6(a, b):
    """
    invalid index type
    :param a:
    :return:
    """
    for u, v in enumerate(a):
        b[v] = u


def test_index_7(a, b, m, n):
    """
    test different slices
    :param a:
    :param b:
    :return:
    """
    for u, v in zip(a[1::m], b[:256:n]):
        print(u, v)
