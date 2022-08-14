def blah(x, y):
    for i, (u, v) in enumerate(zip(x, y)):
        print(i, u, v)
        continue
        for j in range(8):
            print('should not appear')
