import os
import numpy as np

import driver

import type_interface as ti

from errors import CompilerError
from pretty_printing import pretty_printer

tests = ("test_forifcont.py", "test_nested.py", "test_cascade_if.py", "test_dead.py", "test_dead2.py",
         "test_while.py", "test_cascade_assign.py", "test_for.py", "test_forif.py", "test_retval.py",
         "test_pass.py", "test_conditional_terminated.py", "test_bothterminated.py", "test_chained_comparisons.py",
         "test_folding.py", "test_fold_unreachable.py", "test_normalize_return_flow.py", "unpack_with_subscripts.py",
         "test_nested_if.py", "test_nested_if_non_const.py")

array_2d = ti.array_arg_from_spec(ndims=2, dtype=np.float64, fixed_dims=(), evol=None)
array_1d = ti.array_arg_from_spec(ndims=1, dtype=np.float64, fixed_dims=(), evol=None)
int_32 = ti.scalar_type_from_spec(bits=32, is_integral=True, is_boolean=False)
int_64 = ti.scalar_type_from_spec(bits=64, is_integral=True, is_boolean=False)
float_32 = ti.scalar_type_from_spec(bits=32, is_integral=False, is_boolean=False)
float_64 = ti.scalar_type_from_spec(bits=64, is_integral=False, is_boolean=False)
bool_t = ti.scalar_type_from_spec(bits=8, is_integral=True, is_boolean=True)

# Todo: This needs a formal test suite rather than examples.
#       For now, make a second tuple to inject type info and extend
#       the printer to handle operator precedence and generation of annotations.

# Todo: should rewrite variable names so that they pair one to one with types


uws_types = {"test_forifcont.py": {"something":
                                      {"i": int_32, "u": float_64, "v": float_64, "x": array_1d,
                                       "y": array_1d, "z": array_1d}},
             "test_nested.py":
                 {"test_n":
                      {"i": int_32, "j": int_32, "k": int_32, "n": int_32, "m": int_32, "p": int_32}},
             "test_cascade_if.py":
                 {"examp":
                     {"a": int_64, "b": float_64, "c": float_64}},
             "test_dead.py":
                 {"blah":
                     {"a": array_1d, "b": array_1d, "i": float_64, "j": float_64}},
             "test_dead2.py":
                 {"blah":
                     {"a": array_1d, "b": array_1d, "u": float_64, "v": float_64, "c": array_1d}},
             "test_while.py":
                 {"f":
                     {"x": int_32, "i": int_32}},
             "test_cascade_assign.py":
                 {"ca":
                     {"a": int_32, "b":  int_32, "c":  int_32, "d": int_32, "e": int_32, "f":  int_32}},
             "test_for.py":
                 {"for_func":
                     {"i": int_64, "u": float_64, "v": float_64, "x": array_1d,
                      "y": array_1d, "z": array_1d}},
             "test_forif.py":
                 {"something":
                      {"i": int_32, "u": float_64, "v": float_64, "x": array_1d, "y": array_1d, "z": array_1d}},
             "test_retval.py":
                 {"f":
                      {"x": bool_t}},
             "test_pass.py":
                 {"blah":
                      {"i": int_32, "j": int_32, "u": float_64, "v": float_64, "x": array_1d,
                       "y": array_1d}},
             "test_conditional_terminated.py":
                 {"divergent":
                      {"a": array_1d, "b": array_1d,
                       "c": array_1d, "h": float_64, "i": float_64, "j": float_64}},
             "test_bothterminated.py":
                 {"both_term":
                      {"a": int_64, "b": int_64, "c": array_1d, "d": int_32, "e": int_32, "f": array_1d,
                       "g": int_32, "k": float_64, "m": float_64, "n": array_1d}},
             "test_chained_comparisons.py":
                 {"chained_compare":
                      {"a": float_32, "b": float_32, "c": float_32}},
             "test_folding.py":
                 {"folding":
                      {"a": array_1d, "b": array_1d, "c": array_1d, "d": array_1d, "i": int_32}},
             "test_fold_unreachable.py":
                 {"divergent":
                      {"a": array_1d, "b": array_1d, "c": array_1d, "h": float_64,
                       "i": float_64, "j": float_64, "k": int_32}},
             "test_sym.py":
                 {"f":
                      {"x": int_64, "y": int_32}},
             "test_normalize_return_flow.py":
                 {"something":
                      {"a": int_64, "b": int_32}},
             "unpack_with_subscripts.py":
                 {"unpack_test":
                      {"a": array_1d, "b": array_1d, "c": array_1d, "d": array_1d,
                       "i": int_32, "k": float_64, "u": float_64, "v": float_64}},
             "test_nested_if.py":
                 {"nested":
                      {"a": int_64, "b": int_64, "c": int_64, "d": array_1d, "v": float_64}},
             "test_nested_if_non_const.py":
                 {"nested":
                      {"a": float_64, "b": float_64, "c": float_64, "d": array_1d, "v": float_64}}
             }


for i, t in enumerate(tests):
    print(t)
    filepath = os.path.join("tests", t)
    print("\n\nSOURCE\n\n")
    with open(filepath) as reader:
        src = reader.read()
    print(src)
    print("\n\nOUTPUT\n\n")
    types = uws_types.get(t, {})
    try:
        module = driver.compile_module(filepath, types, print_result=True)
    except CompilerError as e:
        msg = f"Error in module: {filepath}"
        raise Exception(msg) from e
    print('\n\n\n')
