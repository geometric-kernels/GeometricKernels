from typing import Union

from spaces.manifold import Manifold
from spaces.mesh import Mesh
from spaces.graph import Graph

Space = Union[Manifold, Mesh, Graph]
