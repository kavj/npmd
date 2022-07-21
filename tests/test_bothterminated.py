def both_term(a, b, c, d, e, f, n):
    g = 0
    if a > b:
        for k in c:
            print(k)
            if k != 2:
                continue
                for m in n:
                    print(m)
            else:
                k += 3
                for m in n:
                    print(m)
                break
            for p in n:
                print("should not appear")
        return g
    else:
        g = 0
        for e in f:
            g += e
        return g
    print("should not appear")
