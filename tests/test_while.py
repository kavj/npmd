# cython: language_level=3, boundscheck=False, wraparound=False, control_flow.dot_output=while.dot


def f(x):
    i = 0
    while i < x:
        i += 1
