

def something(a,b):
    # divergent return with no loop
    if a > b:
        return a
    a *= b
    return a // 2
    if True:
        print("junk statement")