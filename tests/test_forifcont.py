# cython: control_flow.dot_output=blah.dot

import numpy as np


def something(x, y, z):
    for i, (u, v) in enumerate(zip(x, y)):
        if u < 0:
            print(u)
        else:
            print(v)
            continue
        z[i] = u + v
    return 42
