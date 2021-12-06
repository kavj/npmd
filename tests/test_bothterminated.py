def both_term(a, b, c, d, e, f, n):
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
            for m in n:
                print("should not appear")
        return
    else:
        g = 0
        for e in f:
            g += e
        return g
    print("should not appear")
