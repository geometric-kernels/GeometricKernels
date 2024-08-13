"""
Various spaces supported by the library as input domains for kernels.
"""

# noqa: F401
from geometric_kernels.spaces.base import (
    DiscreteSpectrumSpace,
    NoncompactSymmetricSpace,
    Space,
)
from geometric_kernels.spaces.circle import Circle
from geometric_kernels.spaces.graph import Graph
from geometric_kernels.spaces.hyperbolic import Hyperbolic
from geometric_kernels.spaces.hypercube_graph import HypercubeGraph
from geometric_kernels.spaces.hypersphere import Hypersphere
from geometric_kernels.spaces.lie_groups import CompactMatrixLieGroup
from geometric_kernels.spaces.mesh import Mesh
from geometric_kernels.spaces.product import ProductDiscreteSpectrumSpace
from geometric_kernels.spaces.so import SpecialOrthogonal
from geometric_kernels.spaces.spd import SymmetricPositiveDefiniteMatrices
from geometric_kernels.spaces.su import SpecialUnitary
