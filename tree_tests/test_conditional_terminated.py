def divergent(a, b, c):
    for i in a:
        for j in b:
            if not j:
                continue
            print("should be nested")
        for h in c:
            if h > 4:
                break
            print("should also be nested")
