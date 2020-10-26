

def for_func(x,y,z):
    for i, (u,v) in enumerate(zip(x,y)):
         z[i] = u + v
