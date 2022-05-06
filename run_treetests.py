import logging
import pathlib
import numpy as np

import driver
import ir

from errors import CompilerError


tests = ("test_forifcont.py", "test_nested.py", "test_cascade_if.py", "test_dead.py", "test_dead2.py",
         "test_while.py", "test_cascade_assign.py", "test_for.py", "test_forif.py", "test_retval.py",
         "test_pass.py", "test_conditional_terminated.py", "test_bothterminated.py", "test_chained_comparisons.py",
         "test_folding.py", "test_fold_unreachable.py", "test_normalize_return_flow.py", "unpack_with_subscripts.py",
         "test_nested_if.py")
# tests = ("test_conditional_terminated.py",)


array_2d = ir.ArrayType(ndims=2, dtype=ir.float64)
array_1d = ir.ArrayType(ndims=1, dtype=ir.float64)
array_1d_int32 = ir.ArrayType(ndims=1, dtype=ir.int32)


# Todo: This needs a formal test suite rather than examples.
#       For now, make a second tuple to inject type info and extend
#       the printer to handle operator precedence and generation of annotations.

# Todo: should rewrite variable names so that they pair one to one with types


uws_types = {"test_forifcont.py": {"something":
                                       {"i": np.dtype('int32'), "u": np.dtype('float64'), "v": np.dtype('float64'), "x": array_1d,
                                        "y": array_1d, "z": array_1d}},
             "test_nested.py":
                 {"test_n":
                      {"i": np.dtype('int32'), "j": np.dtype('int32'), "k": np.dtype('int32'), "n": np.dtype('int32'), "m": np.dtype('int32'), "p": np.dtype('int32')}},
             "test_cascade_if.py":
                 {"examp":
                      {"a": np.dtype('int64'), "b": np.dtype('float64'), "c": np.dtype('float64')}},
             "test_dead.py":
                 {"blah":
                      {"a": array_1d, "b": array_1d, "i": np.dtype('float64'), "j": np.dtype('float64')}},
             "test_dead2.py":
                 {"blah":
                      {"a": array_1d_int32, "b": array_1d_int32, "u": np.dtype('int32'), "v": np.dtype('int32'), "c": np.dtype('int32')}},
             "test_while.py":
                 {"f":
                      {"x": np.dtype('int32'), "i": np.dtype('int32')}},
             "test_cascade_assign.py":
                 {"ca":
                      {"a": np.dtype('int32'), "b": np.dtype('int32'), "c": np.dtype('int32'), "d": np.dtype('int32'), "e": np.dtype('int32'), "f": np.dtype('int32')}},
             "test_for.py":
                 {"for_func":
                      {"i": np.dtype('int64'), "u": np.dtype('float64'), "v": np.dtype('float64'), "x": array_1d,
                       "y": array_1d, "z": array_1d}},
             "test_forif.py":
                 {"something":
                      {"i": np.dtype('int32'), "u": np.dtype('float64'), "v": np.dtype('float64'), "x": array_1d, "y": array_1d, "z": array_1d}},
             "test_retval.py":
                 {"f":
                      {"x": np.dtype('bool')}},
             "test_pass.py":
                 {"blah":
                      {"i": np.dtype('int32'), "j": np.dtype('int32'), "u": np.dtype('float64'), "v": np.dtype('float64'), "x": array_1d,
                       "y": array_1d}},
             "test_conditional_terminated.py":
                 {"divergent":
                      {"a": array_1d, "b": array_1d,
                       "c": array_1d, "h": np.dtype('float64'), "i": np.dtype('float64'), "j": np.dtype('float64')}},
             "test_bothterminated.py":
                 {"both_term":
                      {"a": np.dtype('int64'), "b": np.dtype('int64'), "c": array_1d, "d": np.dtype('int32'), "e": np.dtype('int32'), "f": array_1d_int32,
                       "g": np.dtype('int32'), "k": np.dtype('float64'), "m": np.dtype('float64'), "n": array_1d}},
             "test_chained_comparisons.py":
                 {"chained_compare":
                      {"a": np.dtype('float32'), "b": np.dtype('float32'), "c": np.dtype('float32')}},
             "test_folding.py":
                 {"folding":
                      {"a": np.dtype('int32'), "b": np.dtype('int64'), "c": np.dtype('int64'), "d": np.dtype('int64'), "i": np.dtype('int32')}},
             "test_fold_unreachable.py":
                 {"divergent":
                      {"a": array_1d, "b": array_1d, "c": array_1d, "h": np.dtype('float64'),
                       "i": np.dtype('float64'), "j": np.dtype('float64'), "k": np.dtype('int32')}},
             "test_sym.py":
                 {"f":
                      {"x": np.dtype('int64'), "y": np.dtype('int32')}},
             "test_normalize_return_flow.py":
                 {"something":
                      {"a": np.dtype('int64'), "b": np.dtype('int32')}},
             "unpack_with_subscripts.py":
                 {"unpack_test":
                      {"a": array_1d, "b": array_1d, "c": array_1d, "d": np.dtype('int32'),
                       "i": np.dtype('float64'), "k": np.dtype('float64'), "u": np.dtype('float64'), "v": np.dtype('float64')}},
             "test_nested_if.py":
                 {"nested":
                      {"a": np.dtype('int64'), "b": np.dtype('int64'), "c": np.dtype('int64'), "d": array_1d, "v": np.dtype('float64')}},
             "test_nested_if_non_const.py":
                 {"nested":
                      {"a": np.dtype('float64'), "b": np.dtype('float64'), "c": np.dtype('float64'), "d": array_1d, "v": np.dtype('float64')}},
             "test_double_nested.py":
                 {"double_nesting":
                      {"s": np.dtype('float64'), "a": array_2d, "b": array_1d, "i": np.dtype('int64'),
                       "sub_a": array_1d, "sub_sub_a": np.dtype('float64'), "sub_b": np.dtype('float64')}
                  }
             }

failed_tests = []

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
        module = driver.compile_module(inpath, types, print_result=True, out_dir=outpath, check_unbound=True)
    except CompilerError as ce:
        msg = f"Failed test: {t}"
        print(ce)
        failed_tests.append(msg)
    print('\n\n\n')

for msg in failed_tests:
    logging.warning(msg=msg)
