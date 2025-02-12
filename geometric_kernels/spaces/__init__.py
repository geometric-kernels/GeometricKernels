"""
Various spaces supported by the library as input domains for kernels.
"""

# noqa: F401
from geometric_kernels.spaces.base import (
    DiscreteSpectrumSpace,
    HodgeDiscreteSpectrumSpace,
    NoncompactSymmetricSpace,
    Space,
)
from geometric_kernels.spaces.circle import Circle
from geometric_kernels.spaces.graph import Graph
from geometric_kernels.spaces.graph_edges import GraphEdges
from geometric_kernels.spaces.hyperbolic import Hyperbolic
from geometric_kernels.spaces.hypercube_graph import HypercubeGraph
from geometric_kernels.spaces.hypersphere import Hypersphere
from geometric_kernels.spaces.lie_groups import CompactMatrixLieGroup
from geometric_kernels.spaces.homogeneous_spaces import CompactHomogeneousSpace
from geometric_kernels.spaces.mesh import Mesh
from geometric_kernels.spaces.product import ProductDiscreteSpectrumSpace
from geometric_kernels.spaces.so import SpecialOrthogonal
from geometric_kernels.spaces.spd import SymmetricPositiveDefiniteMatrices
from geometric_kernels.spaces.su import SpecialUnitary
from geometric_kernels.spaces.stiefel import Stiefel
from geometric_kernels.spaces.grassmannian import Grassmannian

