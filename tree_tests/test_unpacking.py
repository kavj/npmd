
def test_unpack_basic(a, b):
    """
    test raises if not fully unpacked
    :param a:
    :param b:
    :return:
    """
    for u, v in zip(a, b):
        print(u, v)
