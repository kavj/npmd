import ASTBuilder as ab
import ast
from astpretty import pprint


def test():
    with open("simple.py") as src:
        src = src.read()

    tree = ast.parse(src)
