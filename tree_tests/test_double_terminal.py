

def double_continue(a, b):
    for u, v in zip(a, b):
        if u > 0:
            continue
        else:
            continue


def triple_continue(a, b):
    for u, v in zip(a, b):
        if u > 0:
            continue
        elif v < 0:
            continue
        else:
            continue


def incompatible_continue_break_mix(a, b):
    for u, v in zip(a, b):
        if u > 0:
            continue
        elif v < 0:
            break
        else:
            continue


def compatible_single_break_double_continue(a, b):
    for u, v in zip(a, b):
        if u > 0:
            break
        elif v < 0:
            continue
        else:
            continue
