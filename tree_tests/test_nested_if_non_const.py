def nested(a, b, c, d):
    """
    Test correctness edge case of branch inlining.
    """
    for v in d:
        if a < b:
            print(a)
            if b > c:
                if a != c:
                    s = "should remain in branch"
                else:
                    break
                s = "should be appended to true condition"
