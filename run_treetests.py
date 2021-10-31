import os
import numpy as np

import driver

import type_interface as ti

from pretty_printing import pretty_printer

tests = ("test_forifcont.py", "test_nested.py", "test_cascade_if.py", "test_dead.py", "test_dead2.py",
         "test_while.py", "test_cascade_assign.py", "test_for.py", "test_forif.py", "test_retval.py",
         "test_pass.py", "test_conditional_terminated.py", "test_bothterminated.py", "test_chained_comparisons.py",
         "test_folding.py", "test_fold_unreachable.py", "test_normalize_return_flow.py", "unpack_with_subscripts.py",
         "test_nested_if.py", "test_nested_if_non_const.py")

array_2d = ti.array_arg_from_spec(ndims=2, dtype=np.float64, fixed_dims=(), evol=None)
array_1d =  ti.array_arg_from_spec(ndims=1, dtype=np.float64, fixed_dims=(), evol=None)
int_32 = ti.scalar_type_from_spec(bits=32, is_integral=True, is_boolean=False)
int_64 = ti.scalar_type_from_spec(bits=64, is_integral=True, is_boolean=False)
float_64 = ti.scalar_type_from_spec(bits=64, is_integral=False, is_boolean=False)
bool_t = ti.scalar_type_from_spec(bits=8, is_integral=True, is_boolean=True)

# Todo: This needs a formal test suite rather than examples.
#       For now, make a second tuple to inject type info and extend
#       the printer to handle operator precedence and generation of annotations.


uws_types = {"test_forifcont.py": {"something": {"x": array_1d,
                                                 "y": array_1d,
                                                 "z": array_1d}
                                   },
             "test_nested.py": {"test_n": {"n": int,
                                           "m": int,
                                           "p": int}
                                },
             "test_cascade_if.py":
                 {"examp": {"a": int_64, "b": float_64, "c": float_64}},
             "test_dead.py":
                 {"blah": {"a": array_1d, "b": array_1d}},
             "test_dead2.py":
                 {"blah": {"a": array_1d, "b": array_1d}},
             "test_while.py":
                 {"f": {"x": int_32}},
             "test_cascade_assign.py":
                 {"ca": {"d": int_32}},
             "test_for.py":
                 {"for_func": {"i": int_64, "u": float_64, "v": float_64, "x": array_1d,
                                          "y": array_1d, "z": array_1d}},
             "test_forif.py":
                 {"something": {}},
             "test_retval.py":
                 {"f": {"x": bool_t}},
             "test_pass.py":
                 {"blah": {"i": int_32, "j": int_32, "u": float_64, "v": float_64, "x": array_1d,
                           "y": array_1d}},
             "test_conditional_terminated.py":
                 {"divergent": {"a": array_1d, "b": array_1d,
                               "c": array_1d, "h": float_64, "i":float_64, "j":float_64}},
             "test_bothterminated.py":
                 {"both_term": {}},
             "test_chained_comparisons.py":
                 {"chained_compare": {}},
             "test_folding.py":
                 {"folding": {"a": array_1d, "b": array_1d, "c": array_1d, "d": array_1d, "i": int_32}},
             "test_fold_unreachable.py":
                 {"divergent": {"a": array_1d, "b": array_1d, "c": array_1d, "h": float_64,
                                "i": float_64, "j": float_64, "k": int_32}},
             "test_sym.py":
                 {"f": {"x": int_64, "y": int_32}},
             "test_normalize_return_flow.py":
                 {"something": {"a": int_64, "b": int_32}},
             "unpack_with_subscripts.py":
                 {"unpack_test": {"a": array_1d, "b": array_1d, "c": array_1d, "d": array_1d,
                                  "i": int_32, "k": float_64, "u": float_64, "v": float_64}},
             "test_nested_if.py": {"nested":
                                {"a": int_64, "b": int_64, "c": int_64, "d": array_1d}},
             "test_nested_if_non_const.py":
                 {"nested": {"a": float_64, "b": float_64, "c": float_64, "d": array_1d, "v": float_64}}
             }

def divergent(a, b, c):
    for i in a:
        for j in b:
            if not j:
                continue
            print("should be nested")
        for h in c:
            if h > 4:
                break
            print("should also be nested")


Type_Info = {}


for i, t in enumerate(tests):
    print(t)
    filepath = os.path.join("tests", t)
    print("\n\nSOURCE\n\n")
    with open(filepath) as reader:
        src = reader.read()
    print(src)
    print("\n\nOUTPUT\n\n")
    types = uws_types.get(t, {})
    module = driver.compile_module(filepath, types, print_result=True)
    print('\n\n\n')
