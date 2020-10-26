



def test_n(n,m,p):
    for i in range(n):
        for j in range(m):
            print(i,j)
            for k in range(p):
                if p < k:
                    print(p)
                else:
                    return k
