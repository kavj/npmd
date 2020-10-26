import os

from TreePipeline import run_tree_pipeline as rtp
from printing import printtree

tests = (
    "unpack_with_subscripts.py", "test_nested.py", "test_cascade_if.py", "test_dead.py", "test_dead2.py",
    "test_while.py",
    "test_for.py", "test_forif.py", "test_while.py", "test_retval.py", "test_pass.py", "test_conditional_terminated.py",
    "test_bothterminated.py")

# tests = ("test_while.py",)

# tests = ("test_retval.py",)

# tests = ("test_bothterminated.py",)
# tests = ("test_conditional_terminated.py",)

P = printtree()

for i, t in enumerate(tests):
    print(t)
    filepath = os.path.join("tests", t)
    print("\n\nSOURCE\n\n")
    with open(filepath) as reader:
        src = reader.read()
    print(src)
    print("\n\nOUTPUT\n\n")
    mod = rtp(filepath)
    P(mod)
    print('\n\n\n')
