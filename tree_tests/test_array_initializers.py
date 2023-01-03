import numpy as np


def test_array_1d(n):
    """
    tests creation of a single dimension empty array
    :param n:
    :return:
    """
    return np.empty(n)


def test_array_init(n):
    return np.zeros(shape=(n, n))
