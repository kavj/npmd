import ast
from astutils import post_order_nodes as pon

s = "def f(x):\n    for v in x:\n        print(v)"
for node in pon(ast.parse(s)):
    print(node)
