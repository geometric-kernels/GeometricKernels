from typing import Optional
from gpflow.utilities.multipledispatch import Dispatcher

from geometric_kernels.kernels.abstract_kernel import MeshKernel, SphereKernel


BaseGeometricKernel = Dispatcher("BaseGeometricKernel")


@BaseGeometricKernel.register(spaces.manifold.backend.Hypersphere)
def _SphereBaseGeometricKernel(space: spaces.manifold.backend.Hypersphere, *, nu: Optional[float] = None, num_features: Optional[int] = None):
    return SphereKernel(space, nu, num_features)


@BaseGeometricKernel.register(spaces.manifold.backend.Nesh)
def _MeshBaseGeometricKernel(space: spaces.mesh.Mesh, *, nu: Optional[float]=None, num_features: Optional[int]=None):
    # Retrieve Laplace eigendecomposition
    eigenfunctions = None
    eigenvalues = None
    return MeshKernel(space, nu, eigenfunctions, eigenvalues)
