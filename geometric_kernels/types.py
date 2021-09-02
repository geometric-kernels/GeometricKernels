"""
Types used across the package.
"""
from typing import Any

# TODO(VD): check EagerPy for backend agnostic types
# from eagerpy.types import NativeTensor
from eagerpy.types import NativeTensor
from numpy import ndarray

TensorLike = NativeTensor
Parameter = Any
