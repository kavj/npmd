# cython: language_level=3, boundscheck=False, wraparound=False, control_flow.dot_output=dead.dot

def blah(a, b):
    for i in a:
        continue
        for j in b:
            print(j)
