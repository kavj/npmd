

def double_nesting(a, b):
    s = 0.0
    for sub_a in a:
        for i, (sub_sub_a, sub_b) in enumerate(zip(sub_a, b)):
            s += sub_sub_a * sub_b
    return s
