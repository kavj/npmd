

def f(x, y, z):
    for i, (u, v) in enumerate(x, y):
        z[i] = 2 * u + v
        z[i] += 8 // 14

