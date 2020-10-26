
import ast



def test():
    print('something')
    s = "def f(x,y):\n    for i in range(8):\n        x[i] = 5 + y[i]"
    T = ast.parse(s)
    for i, node in enumerate(ast.walk(T)):
        #print(i, type(node))
        if isinstance(node, ast.BinOp):
            print('start')
            for n in ast.iter_fields(node):
                print(n)
            break

