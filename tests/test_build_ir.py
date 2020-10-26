import ASTBuilder as ab
import ast
from astpretty import pprint

def test():
    with open("simple.py") as src:
        src = src.read()

    print(src)
    
    tree = ast.parse(src)     

    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

    #pprint(tree)
    for func in funcs:
        func = ab.NormalizeCalls().visit(func)

    
    pprint(tree)
    print('\n\n\n')

    print('trees down')
    G = ab.build_module_ir(src, {})
    return G