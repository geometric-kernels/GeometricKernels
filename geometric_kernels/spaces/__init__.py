#noqa: F401
from typing import Union

from geometric_kernels.spaces.manifold import Manifold
from geometric_kernels.spaces.mesh import Mesh
from geometric_kernels.spaces.graph import Graph

Space = Union[Manifold, Mesh, Graph]
