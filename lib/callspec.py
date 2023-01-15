import itertools
import numpy

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import lib.ir as ir

from lib.errors import CompilerError

# we should be able to map this to type, return_type or similar


# Don't embed name so that we can reuse an argument signature across many similar functions

@dataclass(frozen=True)
class argsig:
    required_args: Tuple[str, ...]
    default_args: Tuple[Tuple[str, Union[numpy.dtype, ir.ValueRef]], ...] = ()
    required_supports_keywords: bool = False  # some signatures have required positional only arguments

    def __post_init__(self):
        # ensure unique
        assert len(set(self.required_args)) == len(self.required_args) \
               and len(set(self.default_args)) == len(self.default_args)
        assert Ellipsis not in self.default_args
        if Ellipsis in self.required_args:
            assert len(self.required_args) == 1
            assert not self.default_args
            assert not self.required_supports_keywords

        seen = set()
        for arg in self.required_args:
            if arg in seen:
                raise ValueError(f'"{arg}" appears more than once in signature {self}.')
            seen.add(arg)
        for arg, _ in self.default_args:
            if arg in seen:
                raise ValueError(f'"{arg}" appears more than once in signature {self}.')
            seen.add(arg)

    def contains_ellipsis(self):
        return Ellipsis in self.required_args

    def argindex(self, arg: str) -> Optional[int]:
        for index, name in enumerate(self.required_args):
            if name == arg:
                return index
        for index, (name, _) in enumerate(self.default_args, start=len(self.required_args)):
            if name == arg:
                return index

    def all_arg_names(self):
        for arg in self.required_args:
            yield arg
        for arg, _ in self.default_args:
            yield arg

    def is_allowed_keyword(self, argname: str):
        if self.required_supports_keywords:
            for arg in self.required_args:
                if arg == argname:
                    return True
        for arg, _ in self.default_args:
            if arg == argname:
                return True
        return False


# Extending a lot of this stuff requires a signature, describing how they may be called, a transformer
# and a validator.

# Todo: make this loadable from toml

# supported untyped signatures
ellips = argsig((...,))
builtin_reduction = argsig(('iterable',))

builtins = {

    'range': [argsig(('stop',)),
              argsig(('start', 'stop')),
              argsig(('start', 'stop', 'step'))],
    'enumerate': [argsig(('iterable',), (('start', ir.Zero),), True)],
    'len': [argsig(('obj',))],
    'max': [ellips],
    'min': [ellips],
    'zip': [ellips],
    'abs': [argsig(('x',))],
    'all': [builtin_reduction],
    'any': [builtin_reduction]
}


def replace_builtin(node: ir.Call):
    assert not node.kwargs
    if node.func == 'range':
        # range doesn't support keyword arguments, so we can't treat these as defaults
        args = deque(node.args.subexprs)
        if len(args) == 1:
            args.appendleft(ir.Zero)
        if len(args) == 2:
            args.append(ir.One)
        assert len(args) == 3
        return ir.AffineSeq(*args)
    elif node.func == 'enumerate':
        return ir.Enumerate(*node.args.subexprs)
    elif node.func == 'len':
        arg, = node.args.subexprs
        return ir.SingleDimRef(arg, ir.Zero)
    elif node.func == 'zip':
        return ir.Zip(*node.args.subexprs)
    elif node.func == 'abs':
        arg, = node.args.subexprs
        return ir.ABSOLUTE(arg)
    elif node.func == 'all':
        arg, = node.args.subexprs
        return ir.ALL(arg)
    elif node.func == 'any':
        arg, = node.args.subexprs
        return ir.ANY(arg)
    elif node.func == 'max':
        args = deque(node.args.subexprs)
        assert len(args) > 1
        a = args.popleft()
        b = args.popleft()
        terms = ir.MAX(a, b)
        while args:
            b = args.popleft()
            terms = ir.MAX(terms, b)
        return terms
    elif node.func == 'min':
        args = deque(node.args.subexprs)
        assert len(args) > 1
        a = args.popleft()
        b = args.popleft()
        terms = ir.MIN(a, b)
        while args:
            b = args.popleft()
            terms = ir.MIN(terms, b)
        return terms
    else:
        return node


# These have to represent all arguments in the original signature, regardless of whether we support them until
# we won't support any other based on positional assignment
float64 = ir.StringConst('float64')
K = ir.StringConst('K')
same_kind = ir.StringConst('same_kind')
none_ref = ir.NoneRef()

array_init_sig = argsig(('shape',), (('dtype', float64),), True)


binop_sig = argsig(('x1', 'x2',), (('out', none_ref),
                                   ('where', none_ref),
                                   ('casting', same_kind),
                                   ('order', K),
                                   ('dtype', float64)))


unary_sig = argsig(('x1',), (('out', none_ref),
                             ('where', none_ref),
                             ('casting', same_kind),
                             ('order', K),
                             ('dtype', none_ref)))


arithmetic_reduc_sig = argsig(('a',),
                              (('axis', ir.Zero),
                              ('dtype', none_ref),
                              ('out', none_ref),
                              ('keepdims', ir.FALSE),
                              ('initial', none_ref),
                              ('where', none_ref)),
                              True)

logical_reduc_sig = argsig(('a',),
                           (('axis', none_ref),
                           ('out', none_ref),
                           ('keepdims', ir.FALSE),
                           ('where', none_ref)),
                           True)


binops = {'add': ir.ADD,
          'divide': ir.TRUEDIV,
          'floor_divide': ir.FLOORDIV,
          'maximum': ir.MAX,
          'minimum': ir.MIN,
          'multiply': ir.MULT,
          'subtract': ir.SUB,
          'true_divide': ir.TRUEDIV,
          'left_shift': ir.LSHIFT,
          'right_shift': ir.RSHIFT,
          'bitwise_and': ir.BITAND,
          'bitwise_or': ir.BITOR,
          'bitwise_xor': ir.BITXOR,
          'equal': ir.EQ,
          'greater': ir.GT,
          'greater_equal': ir.GE,
          'less': ir.LT,
          'less_equal': ir.LE,
          'not_equal': ir.NE
          }


unaryops = {'abs': ir.ABSOLUTE,
            'absolute': ir.ABSOLUTE,
            'bitwise_not': ir.UINVERT,
            'floor': ir.FLOOR,
            'invert': ir.UINVERT,
            'logical_not': ir.NOT,
            'negative': ir.USUB,
            'square': ir.MULT,
            'sqrt': ir.SQRT,
            }


initializers = {'empty': ir.ArrayInitializer,
                'ones': ir.ArrayInitializer,
                'zeros': ir.ArrayInitializer
                }


logical_reductions = {'all': ir.ALL,
                      'any': ir.ANY
                      }


expr_map = {
    'abs': ir.ABSOLUTE,
    'absolute': ir.ABSOLUTE,
    'add': ir.ADD,
    'divide': ir.TRUEDIV,
    'floor': ir.FLOOR,
    'floor_divide': ir.FLOORDIV,
    'max': ir.MAX,
    'min': ir.MIN,
    'sub': ir.SUB
}


def format_spec(name: str, spec: argsig):
    kwargs = []
    for arg, default_value in spec.default_args:
        if isinstance(default_value, (ir.CONSTANT, ir.StringConst)):
            value = default_value.value
        elif isinstance(default_value, ir.NoneRef):
            value = 'None'
        else:
            value = default_value
        formatted = f'{arg} = {value}'
        kwargs.append(formatted)
    all_args = ', '.join(arg for arg in itertools.chain(spec.required_args, kwargs))
    func = f'{name}({all_args})'
    return func


def map_keywords(node: ir.Call):
    kw_to_value = {}
    for kw_and_value in node.kwargs.subexprs:
        kw, value = kw_and_value.subexprs
        if kw in kw_to_value:
            msg = f'Duplicate keyword "{kw}" in call to "{node.func}".'
            raise CompilerError(msg)
        kw_to_value[kw.name] = value
    return kw_to_value


def try_match_arg_spec(name: str, args: List[ir.ValueRef], kwargs: Dict[str, ir.ValueRef], spec: argsig):
    if spec.contains_ellipsis():
        if kwargs:
            return
        return ir.Call(name, ir.TUPLE(*args), ir.TUPLE(*kwargs))
    args = list(args)
    nargs = len(args)
    nrequired = len(spec.required_args)
    all_arg_names = deque(spec.all_arg_names())
    # get the ones we need
    for _ in args:
        # remove any args that are satisfied by positional arguments
        if not all_arg_names:
            # spec doesn't provide for enough arguments
            return
        all_arg_names.popleft()
    if any(k not in all_arg_names for k in kwargs.keys()):
        return
    if len(args) < len(spec.required_args):
        if spec.required_supports_keywords:
            for arg in itertools.islice(spec.required_args, nargs, nrequired):
                value = kwargs.get(arg)
                if value is None:
                    # cannot map
                    return
                args.append(value)
        else:
            # cannot accommodate
            return
    for kw, default_value in spec.default_args:
        value = kwargs.get(kw, default_value)
        if value is None:
            return
        args.append(value)
    call_args = ir.TUPLE(*args)
    call_kwargs = ir.TUPLE()
    return ir.Call(name, call_args, call_kwargs)


def try_refactor_keywords(node: ir.Call, spec: argsig):
    if not node.kwargs:
        return node
    name = node.func
    args = list(node.args.subexprs)
    kw_to_value = map_keywords(node)
    return try_match_arg_spec(name, args, kw_to_value, spec)


def early_call_specialize(node: ir.Call):
    if node.func in builtins:
        specs = builtins[node.func]
        args = list(node.args.subexprs)
        kw_to_value = map_keywords(node)
        for spec in specs:
            repl_call = try_match_arg_spec(node.func, args, kw_to_value, spec)
            if repl_call is not None:
                repl = replace_builtin(repl_call)
                return repl
    return node
