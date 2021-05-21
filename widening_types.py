"""

Descriptions for inputs and widened inputs

This should contain some kind of SCEV class, for cases where we can represent an integer in this manner

"""
import ir


class scev:

    def __init__(self, name, scalar_type, stride=None):
        self.name = name
        self.stride = stride
        self.scalar_type = scalar_type

    @property
    def constant(self):
        return self.stride == ir.IntNode(0)

    @property
    def unit_stride(self):
        return self.stride == ir.IntNode(1)

    @property
    def indirect_ref(self):
        return self.stride is None


class array_argument:
    """
    distinct from a normal array class in that we want to flatten access patterns across calls so that
    sliding window types and iteration over array dimensions look the same to a function body. Typically
    in optimizable cases, both of these reduce to fixed strides, and we just determine whether gathers are required.

    """

    def __init__(self, name, scalar_type, ndims, evol, uniform_len=True, convertible_to=()):
        self.name = name
        self.scalar_type = scalar_type
        self.ndims = ndims
        self.evol = evol  # scev parameter indicating the start of the array
        self.uniform_len = uniform_len
        self.convertible_to = convertible_to


class vector_type_desc:
    """
    These are settings for a particular vector type, to be specialized for any given
    ISA we are compiling for. The options determine whether each op is allowed.
    If we have disallowed ops, we scalarize the dependencies by promoting varying scalars to arrays
    and varying k dimensional local arrays to k+1 dimensional arrays

    """

    def __init__(self,
                 name,
                 scalar_type,
                 allow_predicate=False,
                 contract_multiply_add=False,
                 allow_unsafe_identity_transforms=False,  # these tend to assume a value is not nan
                 allow_gather=False,
                 allow_scatter=False):
        self.name = name
        self.scalar_type = scalar_type
        self.allow_predicate = allow_predicate
        self.contract_multiply_add = contract_multiply_add
        self.allow_unsafe_identity_transforms = allow_unsafe_identity_transforms
        self.allow_gather = allow_gather
        self.allow_scatter = allow_scatter
