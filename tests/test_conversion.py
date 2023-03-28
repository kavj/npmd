import pathlib
import pytest

# Todo: this could be made into a more complete class
from lib.cfg_builder import build_module_ir
from lib.errors import CompilerError
from tests.type_info import type_detail


# Todo: This needs a formal test suite rather than examples.
#       For now, make a second tuple to inject type info and extend
#       the printer to handle operator precedence and generation of annotations.

unconvertible = () # ('test_unpacking.py',)


def test_conversions():
    basepath = pathlib.Path(__file__).resolve().parent.parent.joinpath('tree_tests')
    for t in basepath.iterdir():
        if t.is_dir():
            # typically just __pycache__
            continue
        file_path = basepath.joinpath(t)
        func_types = type_detail[t.name]
        if file_path.name in unconvertible:
            with pytest.raises(CompilerError):
                build_module_ir(file_path, func_types)
            continue
        if file_path.name != 'test_annotations.py':
            continue
        mod = build_module_ir(file_path, func_types)
        # Now test that conversions yield all nodes
        func = mod.functions[0]
        print(mod.name, func.name)
