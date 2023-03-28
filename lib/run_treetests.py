import logging
import os
import pathlib

import lib.driver as driver
import lib.ir as ir

from lib.errors import CompilerError
from tests.type_info import type_detail as uws_types


tests = ('test_forifcont.py',
         'test_nested.py',
         'test_cascade_if.py',
         'test_dead.py',
         'test_dead2.py',
         'test_while.py',
         'test_cascade_assign.py',
         'test_for.py',
         'test_forif.py',
         'test_retval.py',
         'test_pass.py',
         'test_conditional_terminated.py',
         'test_bothterminated.py',
         'test_chained_comparisons.py',
         'test_folding.py',
         'test_fold_unreachable.py',
         'test_normalize_return_flow.py',
         'test_unpack_with_subscripts.py',
         'test_unpacking.py',
         'test_nested_if.py',
         'test_array_initializers.py',
         'test_double_terminal.py',
         'test_annotations.py')

array_2d = ir.ArrayType(ndims=2, dtype=ir.float64)
array_1d = ir.ArrayType(ndims=1, dtype=ir.float64)
array_1d_int32 = ir.ArrayType(ndims=1, dtype=ir.int32)


# Todo: This needs a formal test suite rather than examples.
#       For now, make a second tuple to inject type info and extend
#       the printer to handle operator precedence and generation of annotations.

# Todo: should rewrite variable names so that they pair one to one with types


failed_tests = []

for i, t in enumerate(tests):
    print(t)
    basepath = pathlib.Path(__file__).resolve().parent.parent.joinpath('tree_tests')
    inpath = basepath.joinpath(t)
    basename, _ = os.path.splitext(t)
    outpath = basepath.joinpath(f"{basename}_{i}")
    print("\n\nSOURCE\n\n")
    with open(inpath) as reader:
        src = reader.read()
    print(src)
    print("\n\nOUTPUT\n\n")
    types = uws_types.get(t, {})
    try:
        module = driver.compile_module(inpath, types, print_result=True, out_dir=outpath, debug=True)
    except CompilerError as ce:
        msg = f"Failed test: {t}: {ce.args[0]}"
        failed_tests.append(msg)
    print('\n\n\n')

for msg in failed_tests:
    logging.warning(msg=msg)
