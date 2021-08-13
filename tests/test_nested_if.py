def nested(a, b, c, d):
    """
    Test correctness edge case of branch inlining.
    """
    for v in d:
        if a < b:
            print(a)
            if b > c:
                if True:
                    print("should be inlined")
                else:
                    break
                print("should also be inlined")
