import os
import numpy as np

import ir
import type_resolution as tr
from TreePipeline import run_tree_pipeline as rtp
from printing import printtree
from ir import ArrayRef, One

# tests = ("test_conditional_terminated.py",)

tests = ("test_forifcont.py", "test_nested.py", "test_cascade_if.py", "test_dead.py", "test_dead2.py")
# "test_while.py", "test_cascade_assign.py", "test_for.py", "test_forif.py", "test_while.py", "test_retval.py",
# "test_pass.py", "test_conditional_terminated.py", "test_bothterminated.py", "test_chained_comparisons.py",
# "test_folding.py", "test_fold_unreachable.py", "test_normalize_return_flow.py")

array_n = ArrayRef(ir.NameRef("blah"), (ir.NameRef("n"),), np.float64)

# uws_types = {"unpack_test": {"a": array_n,
#                             "b": array_n,
#                             "c": array_n,
#                             "d": array_n,
#                             "k": int,
#                             "i": int,
#                             "u": float,
#                             "v": float
#                             }
#             }


# Todo: This needs a formal test suite rather than examples.
#       For now, make a second tuple to inject type info and extend
#       the printer to handle operator precedence and generation of annotations.

uws_types = ({"something": {"x": array_n,
                            "y": array_n,
                            "z": array_n},
              "test_n": {"n": int,
                         "m": int,
                         "p": int}
              })

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
