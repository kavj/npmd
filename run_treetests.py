import pathlib
import numpy as np

import driver

import ir as tr
import type_resolution as tres

from errors import CompilerError

tests = ("test_forifcont.py", "test_nested.py", "test_cascade_if.py", "test_dead.py", "test_dead2.py",
         "test_while.py", "test_cascade_assign.py", "test_for.py", "test_forif.py", "test_retval.py",
         "test_pass.py", "test_conditional_terminated.py", "test_bothterminated.py", "test_chained_comparisons.py",
         "test_folding.py", "test_fold_unreachable.py", "test_normalize_return_flow.py", "unpack_with_subscripts.py",
         "test_nested_if.py", "test_nested_if_non_const.py", "test_double_nested.py")

array_2d = tres.array_arg_from_spec(ndims=2, dtype=np.float64, fixed_dims=())
array_1d = tres.array_arg_from_spec(ndims=1, dtype=np.float64, fixed_dims=())
array_1d_int32 = tres.array_arg_from_spec(ndims=1, dtype=np.int32, fixed_dims=())

# Todo: This needs a formal test suite rather than examples.
#       For now, make a second tuple to inject type info and extend
#       the printer to handle operator precedence and generation of annotations.

# Todo: should rewrite variable names so that they pair one to one with types


uws_types = {"test_forifcont.py": {"something":
                                       {"i": tr.Int32, "u": tr.Float64, "v": tr.Float64, "x": array_1d,
                                        "y": array_1d, "z": array_1d}},
             "test_nested.py":
                 {"test_n":
                      {"i": tr.Int32, "j": tr.Int32, "k": tr.Int32, "n": tr.Int32, "m": tr.Int32, "p": tr.Int32}},
             "test_cascade_if.py":
                 {"examp":
                      {"a": tr.Int64, "b": tr.Float64, "c": tr.Float64}},
             "test_dead.py":
                 {"blah":
                      {"a": array_1d, "b": array_1d, "i": tr.Float64, "j": tr.Float64}},
             "test_dead2.py":
                 {"blah":
                      {"a": array_1d_int32, "b": array_1d_int32, "u": tr.Int32, "v": tr.Int32, "c": tr.Int32}},
             "test_while.py":
                 {"f":
                      {"x": tr.Int32, "i": tr.Int32}},
             "test_cascade_assign.py":
                 {"ca":
                      {"a": tr.Int32, "b": tr.Int32, "c": tr.Int32, "d": tr.Int32, "e": tr.Int32, "f": tr.Int32}},
             "test_for.py":
                 {"for_func":
                      {"i": tr.Int64, "u": tr.Float64, "v": tr.Float64, "x": array_1d,
                       "y": array_1d, "z": array_1d}},
             "test_forif.py":
                 {"something":
                      {"i": tr.Int32, "u": tr.Float64, "v": tr.Float64, "x": array_1d, "y": array_1d, "z": array_1d}},
             "test_retval.py":
                 {"f":
                      {"x": tr.BoolType}},
             "test_pass.py":
                 {"blah":
                      {"i": tr.Int32, "j": tr.Int32, "u": tr.Float64, "v": tr.Float64, "x": array_1d,
                       "y": array_1d}},
             "test_conditional_terminated.py":
                 {"divergent":
                      {"a": array_1d, "b": array_1d,
                       "c": array_1d, "h": tr.Float64, "i": tr.Float64, "j": tr.Float64}},
             "test_bothterminated.py":
                 {"both_term":
                      {"a": tr.Int64, "b": tr.Int64, "c": array_1d, "d": tr.Int32, "e": tr.Int32, "f": array_1d_int32,
                       "g": tr.Int32, "k": tr.Float64, "m": tr.Float64, "n": array_1d}},
             "test_chained_comparisons.py":
                 {"chained_compare":
                      {"a": tr.Float32, "b": tr.Float32, "c": tr.Float32}},
             "test_folding.py":
                 {"folding":
                      {"a": tr.Int32, "b": tr.Int64, "c": tr.Int64, "d": tr.Int64, "i": tr.Int32}},
             "test_fold_unreachable.py":
                 {"divergent":
                      {"a": array_1d, "b": array_1d, "c": array_1d, "h": tr.Float64,
                       "i": tr.Float64, "j": tr.Float64, "k": tr.Int32}},
             "test_sym.py":
                 {"f":
                      {"x": tr.Int64, "y": tr.Int32}},
             "test_normalize_return_flow.py":
                 {"something":
                      {"a": tr.Int64, "b": tr.Int32}},
             "unpack_with_subscripts.py":
                 {"unpack_test":
                      {"a": array_1d, "b": array_1d, "c": array_1d, "d": tr.Int32,
                       "i": tr.Float64, "k": tr.Float64, "u": tr.Float64, "v": tr.Float64}},
             "test_nested_if.py":
                 {"nested":
                      {"a": tr.Int64, "b": tr.Int64, "c": tr.Int64, "d": array_1d, "v": tr.Float64}},
             "test_nested_if_non_const.py":
                 {"nested":
                      {"a": tr.Float64, "b": tr.Float64, "c": tr.Float64, "d": array_1d, "v": tr.Float64}},
             "test_double_nested.py":
                 {"double_nesting":
                      {"s": tr.Float64, "a": array_2d, "b": array_1d, "i": tr.Int64,
                       "sub_a": array_1d, "sub_sub_a": tr.Float64, "sub_b": tr.Float64}
                  }
             }

cannot_compile = []

for i, t in enumerate(tests):
    print(t)
    basepath = pathlib.Path("tests")
    inpath = basepath.joinpath(t)
    outpath = basepath.joinpath(f"test_{i}")
    print("\n\nSOURCE\n\n")
    with open(inpath) as reader:
        src = reader.read()
    print(src)
    print("\n\nOUTPUT\n\n")
    types = uws_types.get(t, {})
    try:
        module = driver.compile_module(inpath, types, print_result=True, out_dir=outpath, check_unbound=False)
    except CompilerError as ce:
        msg = f"{inpath}: {ce.args[0]}"
        cannot_compile.append(inpath.name)

    print('\n\n\n')

if cannot_compile:
    msg = f"Some tests failed: {cannot_compile}"
    raise RuntimeError(msg)
