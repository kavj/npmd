# cython: language_level=3, boundscheck=False, wraparound=False, control_flow.dot_output=dead2.dot


def blah(a, b):
    for u, v in zip(a, b):
        break
        # dead code
        print(u)
        print(v)
        c = v - u
    for v in range(53):
        continue
        # dead code
        return v
        v *= 2
