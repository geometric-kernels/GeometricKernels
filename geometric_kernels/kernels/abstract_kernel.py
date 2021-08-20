import abc
from typing import Callable, List

from geometric_kernels.spaces import Mesh, Space

GeometricKernel = Dispatcher("GeometricKernel")


class BaseGeometricKernel(abc.ABC):
    def __init__(self, space: Space, *args, **kwargs):
        self._space = space
        # self.nu = nu

    @property
    def space(self):
        return self._space

    @abc.abstractmethod
    def K(self, X, X2=None, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def K_diag(self, X, **kwargs):
        raise NotImplementedError


class MeshKernel(BaseGeometricKernel):
    def __init__(
        self, space: Mesh, nu: float, eigenfunctions: Callable, eigenvalues: List
    ):  # TODO:
        super().__init__(space, nu)
        self._num_features = len(eigenvalues)
        self._eigenfunctions = eigenfunctions
        self._eigenvalues = eigenvalues

    def K(self, X, X2=None, **kwargs):
        """Compute the mesh kernel via Laplace eigedecomposition"""
        assert "lengthscale" in kwargs
        pass

    def K_diag(self, X, **kwargs):
        assert "lengthscale" in kwargs
        pass


# class SphereKernel(AbstractGeometricKernel):
#     def __init__(self, space: spaces.manifold.backend.Hypersphere, nu: float, num_features=250):
#         super().__init__(space, nu)
#         self._dim = space.dim
#         self._num_features = num_features

#     def K(self, lengthscale, X, X2=None):
#         """Compute the sphere kernel via Gegenbauer polynomials"""
#         pass

#     def K_diag(self, lengthscale, X):
#         pass

#     def Kchol(self, lengthscale, X, X2=None):
#         pass
