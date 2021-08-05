# cython: control_flow.dot_output=blah2.dot

import numpy as np


def something(x, y, z):
    for i, (u, v) in enumerate(zip(x, y)):
        if u < 0:
            print(u)
        else:
            print(v)
            i += 10
            u *= 2
    return 42
