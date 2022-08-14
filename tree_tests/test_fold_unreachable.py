def divergent(a, b, c):
    k = 0
    for i in a:
        for j in b:
            if not j:
                continue
            print("should be nested")
        for h in c:
            if k > 4:
                break
            print("should be alone in the loop")
