# cython: language_level=3, boundscheck=False, wraparound=False, control_flow.dot_output=blah.dot

def for_func(x):
    n, m, p = x.shape
    for i, x0 in enumerate(x):
        for j, x1 in enumerate(x0):
            for k, x2 in enumerate(x1):
                x[i, j, k] = x1 + 42
