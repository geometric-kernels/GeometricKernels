"""
Spaces on which to define kernels
"""

# noqa: F401
from geometric_kernels.spaces.base import (
    DiscreteSpectrumSpace,
    NoncompactSymmetricSpace,
    Space,
)
from geometric_kernels.spaces.circle import Circle
from geometric_kernels.spaces.graph import Graph
from geometric_kernels.spaces.grassmannian import Grassmannian
from geometric_kernels.spaces.homogeneous_spaces import CompactHomogeneousSpace
from geometric_kernels.spaces.hyperbolic import Hyperbolic
from geometric_kernels.spaces.hypersphere import Hypersphere
from geometric_kernels.spaces.lie_groups import MatrixLieGroup
from geometric_kernels.spaces.mesh import Mesh
from geometric_kernels.spaces.product import ProductDiscreteSpectrumSpace
from geometric_kernels.spaces.so import SpecialOrthogonal
from geometric_kernels.spaces.spd import SymmetricPositiveDefiniteMatrices
from geometric_kernels.spaces.stiefel import Stiefel
from geometric_kernels.spaces.su import SpecialUnitary
