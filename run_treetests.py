import logging
import pathlib
import numpy as np

import driver
import ir

from errors import CompilerError


tests = ("test_forifcont.py", "test_nested.py", "test_cascade_if.py", "test_dead.py", "test_dead2.py",
         "test_while.py", "test_cascade_assign.py", "test_for.py", "test_forif.py", "test_retval.py",
         "test_pass.py", "test_conditional_terminated.py", "test_bothterminated.py", "test_chained_comparisons.py",
         "test_folding.py", "test_fold_unreachable.py", "test_normalize_return_flow.py", "test_unpack_with_subscripts.py",
         "test_nested_if.py")


array_2d = ir.ArrayType(ndims=2, dtype=ir.float64)
array_1d = ir.ArrayType(ndims=1, dtype=ir.float64)
array_1d_int32 = ir.ArrayType(ndims=1, dtype=ir.int32)


# Todo: This needs a formal test suite rather than examples.
#       For now, make a second tuple to inject type info and extend
#       the printer to handle operator precedence and generation of annotations.

# Todo: should rewrite variable names so that they pair one to one with types


uws_types = {"test_forifcont.py": {"something": {"x": array_1d, "y": array_1d, "z": array_1d}},
             "test_nested.py":
                 {"test_n":
                      {"n": np.dtype('int32'), "m": np.dtype('int32'), "p": np.dtype('int32')}},
             "test_cascade_if.py":
                 {"examp":
                      {"a": np.dtype('int64'), "b": np.dtype('float64'), "c": np.dtype('float64')}},
             "test_dead.py":
                 {"blah":
                      {"a": array_1d, "b": array_1d}},
             "test_dead2.py":
                 {"blah":
                      {"a": array_1d_int32, "b": array_1d_int32, "u": np.dtype('int32'), "v": np.dtype('int32'), "c": np.dtype('int32')}},
             "test_while.py":
                 {"f":
                      {"x": np.dtype('int32')}},
             "test_cascade_assign.py":
                 {"ca": {"d": np.dtype('int32')}},
             "test_for.py":
                 {"for_func":
                      { "x": array_1d, "y": array_1d, "z": array_1d}},
             "test_forif.py":
                 {"something": {"x": array_1d, "y": array_1d, "z": array_1d}},
             "test_retval.py":
                 {"f": {"x": np.dtype('bool')}},
             "test_pass.py":
                 {"blah": { "x": array_1d, "y": array_1d}},
             "test_conditional_terminated.py":
                 {"divergent": {"a": array_1d, "b": array_1d, "c": array_1d}},
             "test_bothterminated.py":
                 {"both_term":
                      {"a": np.dtype('int64'), "b": np.dtype('int64'), "c": array_1d, "d": np.dtype('int32'), "e": np.dtype('int32'), "f": array_1d_int32, "n": array_1d}},
             "test_chained_comparisons.py":
                 {"chained_compare": {"a": np.dtype('float32'), "b": np.dtype('float32'), "c": np.dtype('float32')}},
             "test_folding.py":
                 {"folding": {}},
             "test_fold_unreachable.py":
                 {"divergent": {"a": array_1d, "b": array_1d, "c": array_1d, }},
             "test_sym.py":
                 {"f": {"x": np.dtype('int64'), "y": np.dtype('int32')}},
             "test_normalize_return_flow.py":
                 {"something": {"a": np.dtype('int64'), "b": np.dtype('int32')}},
             "test_unpack_with_subscripts.py":
                 {"unpack_test":
                      {"a": array_1d, "b": array_1d, "c": array_1d, "d": np.dtype('int32')}},
             "test_nested_if.py":
                 {"nested":
                      {"a": np.dtype('int64'), "b": np.dtype('int64'), "c": np.dtype('int64'), "d": array_1d}},
             "test_nested_if_non_const.py":
                 {"nested":
                      {"a": np.dtype('float64'), "b": np.dtype('float64'), "c": np.dtype('float64'), "d": array_1d}},
             "test_double_nested.py":
                 {"double_nesting":
                      {"a": array_2d, "b": array_1d}
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
        module = driver.compile_module(inpath, types, print_result=True, out_dir=outpath, ignore_unbound=True)
    except CompilerError as ce:
        msg = f"Failed test: {t}: {ce.args[0]}"
        failed_tests.append(msg)
    print('\n\n\n')

for msg in failed_tests:
    logging.warning(msg=msg)
