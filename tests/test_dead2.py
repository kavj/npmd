# cython: language_level=3, boundscheck=False, wraparound=False, control_flow.dot_output=dead2.dot


def blah(a, b):
    for u, v in zip(a, b):
        break
        print('should be removed')
        print(u)
        print(v)
        c = a - b
    for v in range(53):
        continue
        print('should also be removed')
