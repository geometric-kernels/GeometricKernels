# noqa: F401
from typing import Union

from geometric_kernels.spaces.graph import Graph
from geometric_kernels.spaces.manifold import Manifold
from geometric_kernels.spaces.mesh import Mesh

Space = Union[Manifold, Mesh, Graph]
