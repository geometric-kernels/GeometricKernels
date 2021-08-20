from kernels.abstract_kernel import AbstractGeometricKernel
from kernels.abstract_kernel import SphereKernel
from kernels.abstract_kernel import MeshKernel
from spaces.abstract_space import Space
import spaces.manifold
import spaces.mesh
import spaces.graph

from multipledispatch import dispatch


@dispatch(spaces.manifold.backend.Hypersphere, float, int)
def BaseGeometricKernel(space: spaces.manifold.backend.Hypersphere,
                        nu: float,
                        num_features: int):
    return SphereKernel(space, nu, num_features)


@dispatch(spaces.mesh.Mesh, float, int)
def BaseGeometricKernel(space: spaces.mesh.Mesh,
                        nu: float,
                        num_features: int):
    # Retrieve Laplace eigendecomposition
    eigenfunctions = None
    eigenvalues = None
    return MeshKernel(space, nu, eigenfunctions, eigenvalues)
