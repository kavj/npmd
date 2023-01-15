import numpy as np

import lib.ir as ir

array_1d = ir.ArrayType(ndims=1, dtype=ir.float64)
array_1d_int32 = ir.ArrayType(ndims=1, dtype=ir.int32)
array_2d = ir.ArrayType(ndims=2, dtype=ir.float64)


type_detail = {"test_forifcont.py":
               {"something": {"x": array_1d,
                              "y": array_1d,
                              "z": array_1d}},

               "test_nested.py":
               {"test_n": {"n": np.dtype('int32'),
                           "m": np.dtype('int32'),
                           "p": np.dtype('int32')}},

               "test_cascade_if.py":
               {"examp": {"a": np.dtype('int64'),
                          "b": np.dtype('float64'),
                          "c": np.dtype('float64')}},

               "test_dead.py":
               {"blah": {"a": array_1d,
                         "b": array_1d}},

               "test_dead2.py":
               {"blah": {"a": array_1d_int32,
                         "b": array_1d_int32,
                         "u": np.dtype('int32'),
                         "v": np.dtype('int32'),
                         "c": np.dtype('int32')}},

               "test_while.py":
               {"f": {"x": np.dtype('int32')}},

               "test_cascade_assign.py":
               {"ca": {"d": np.dtype('int32')}},

               "test_for.py":
               {"for_func": {"x": array_1d,
                             "y": array_1d,
                             "z": array_1d}},

               "test_forif.py":
               {"something": {"x": array_1d,
                              "y": array_1d,
                              "z": array_1d}},

               "test_retval.py":
               {"f": {"x": np.dtype('bool')}},

               "test_pass.py":
               {"blah": {"x": array_1d,
                         "y": array_1d}},

               "test_conditional_terminated.py":
               {"divergent": {"a": array_1d,
                              "b": array_1d,
                              "c": array_1d}},

               "test_bothterminated.py":
               {"both_term": {"a": np.dtype('int64'),
                              "b": np.dtype('int64'),
                              "c": array_1d,
                              "d": np.dtype('int32'),
                              "e": np.dtype('int32'),
                              "f": array_1d_int32,
                              "n": array_1d}},

               "test_chained_comparisons.py":
               {"chained_compare": {"a": np.dtype('float32'),
                                    "b": np.dtype('float32'),
                                    "c": np.dtype('float32')}},

               "test_folding.py":
               {"folding": {}},

               "test_fold_unreachable.py":
               {"divergent": {"a": array_1d,
                              "b": array_1d,
                              "c": array_1d, }},

               "test_normalize_return_flow.py":
               {"something": {"a": np.dtype('int64'),
                              "b": np.dtype('int32')}},

               "test_unpack_with_subscripts.py":
               {"unpack_test": {"a": array_1d,
                                "b": array_1d,
                                "c": array_1d,
                                "d": np.dtype('int32')}},

               "test_nested_if.py":
               {"nested": {"a": np.dtype('int64'),
                           "b": np.dtype('int64'),
                           "c": np.dtype('int64'),
                           "d": array_1d}},

               "test_nested_if_non_const.py":
               {"nested": {"a": np.dtype('float64'),
                           "b": np.dtype('float64'),
                           "c": np.dtype('float64'),
                           "d": array_1d}},

               "test_double_nested.py":
               {"double_nesting": {"a": array_2d,
                                   "b": array_1d}},

               "test_branch_nesting.py":
               {
                   'nested': {
                       'a': np.dtype('float64'),
                       'b': np.dtype('float64'),
                       'c': np.dtype('float64'),
                       'd': np.dtype('float64')
                   }
               },

               "test_indexing.py": {
                   'test_index_0':
                       {
                           'a': array_1d
                       },
                   'test_index_1':
                       {
                           'a': array_1d,
                           'n': np.dtype('int32')
                       },
                   'test_index_2':
                       {
                           'a': array_1d,
                           'i': np.dtype('int32')
                       },
                   'test_index_3':
                       {
                           'a': array_1d,
                       },
                   'test_index_4':
                       {
                           'a': array_1d,
                           'i': np.dtype('int32')
                       },
                   'test_index_5':
                       {
                           'a': array_1d,
                           'b': array_1d,
                           'k': np.dtype('int64')
                       },
                   'test_index_6':
                       {
                           'a': array_1d,
                           'b': array_1d,
                       },
                   'test_index_7':
                       {
                           'a': array_1d,
                           'b': array_1d,
                           'm': np.dtype('int32'),
                           'n': np.dtype('int32')
                       }

               },

               'test_unpacking.py': {
                   'test_unpack_basic': {
                       'a': array_1d,
                       'b': array_1d
                   }
               },

               'test_array_initializers.py': {
                   'test_array_1d': {
                       'n': np.dtype('int32')
                   },
                   'test_array_init': {
                       'n': np.dtype('int32')
                   }
               },

               'test_double_terminal.py': {
                   'double_continue': {
                       'a': array_1d,
                       'b': array_1d,
                       'u': np.dtype('float64'),
                       'v': np.dtype('float64')
                   },
                   'triple_continue': {
                       'a': array_1d,
                       'b': array_1d,
                       'u': np.dtype('float64'),
                       'v': np.dtype('float64')
                   },
                   'incompatible_continue_break_mix': {
                       'a': array_1d,
                       'b': array_1d,
                       'u': np.dtype('float64'),
                       'v': np.dtype('float64')
                   },
                   'compatible_single_break_double_continue': {
                       'a': array_1d,
                       'b': array_1d,
                       'u': np.dtype('float64'),
                       'v': np.dtype('float64')
                   }
               }

               }
