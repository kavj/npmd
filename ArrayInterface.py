

class UniformArrayInput:
    """
    Array type descriptor.

    dtype: scalar type used by this array
    dims: tuple(str or int,...)
    stride: input stride

    """

    def __init__(self, dtype, dims):
        self.dtype = dtype
        self.dims = dims


class SlidingWindowInput:
    def __init__(self, dtype, dims, stride):
        self.dtype = dtype
        self.dims = dims
        self.stride = stride


class ByDimArrayInput:
    """
    Iterates acrosss consecutive calls over the leading array dim.

    """

    def __init__(self, dtype, dims):
        self.dtype = dtype
        self.dims = dims

