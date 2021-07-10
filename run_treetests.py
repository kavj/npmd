import os

from ArrayInterface import UniformArrayInput
from TreePipeline import run_tree_pipeline as rtp
from printing import printtree

# tests = ("test_conditional_terminated.py",)

tests = ("unpack_with_subscripts.py",)

# "test_nested.py", "test_cascade_if.py", "test_dead.py", "test_dead2.py",
# "test_while.py", "test_cascade_assign.py", "test_for.py", "test_forif.py", "test_while.py", "test_retval.py",
# "test_pass.py", "test_conditional_terminated.py", "test_bothterminated.py", "test_chained_comparisons.py",
# "test_folding.py", "test_fold_unreachable.py", "test_normalize_return_flow.py")


uws_types = {"unpack_test": {UniformArrayInput(float, ("n",)): {"a", "b", "c", "d"},
                             int: {"k", "i"},
                             float: {"u", "v"}}
             }

Type_Info = {}

P = printtree()

for i, t in enumerate(tests):
    print(t)
    filepath = os.path.join("tests", t)
    print("\n\nSOURCE\n\n")
    with open(filepath) as reader:
        src = reader.read()
    print(src)
    print("\n\nOUTPUT\n\n")
    mod = rtp(filepath, uws_types)
    P(mod)
    print('\n\n\n')
