def nested(a, b, c, d):
    """
    Test correctness edge case of branch inlining.
    """
    for v in d:
        if a < b:
            print(a)
        elif b > c:
            print(b)
        elif d < c:
            print("should be inlined")
        else:
            break
