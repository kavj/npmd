def divergent(a, b, c):
    for u in a:
        for v in b:
            if not v:
                continue
            print("should be nested")
        for h in c:
            if h > 4:
                break
            print("should also be nested")
