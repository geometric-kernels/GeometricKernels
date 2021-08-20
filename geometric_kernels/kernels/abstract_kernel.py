from abc import abstractmethod
from typing import Callable, List

from spaces.abstract_space import Space
import spaces.manifold
import spaces.mesh
import spaces.graph


class AbstractGeometricKernel():
    def __init__(self,
                 space: Space,
                 nu: float,
                 *args, **kwargs):
        self._space = space
        self.nu = nu

    @property
    def space(self):
        return self._space

    @abstractmethod
    def K(self, lengthscale, X, X2=None):
        raise NotImplementedError

    @abstractmethod
    def K_diag(self, lengthscale, X):
        raise NotImplementedError

    @abstractmethod
    def Kchol(self, lengthscale, X, X2=None):
        raise NotImplementedError


class SphereKernel(AbstractGeometricKernel):
    def __init__(self,
                 space: spaces.manifold.backend.Hypersphere,
                 nu: float,
                 num_features=250):
        super().__init__(space, nu)
        self._dim = space.dim
        self._num_features = num_features

    def K(self, lengthscale, X, X2=None):
        """ Compute the sphere kernel via Gegenbauer polynomials """
        pass

    def K_diag(self, lengthscale, X):
        pass

    def Kchol(self, lengthscale, X, X2=None):
        pass


class MeshKernel(AbstractGeometricKernel):
    def __init__(self,
                 space: spaces.mesh.Mesh,
                 nu: float,
                 eigenfunctions: Callable,
                 eigenvalues: List):   # TODO:
        super().__init__(space, nu)
        self._num_features = len(eigenvalues)
        self._eigenfunctions = eigenfunctions
        self._eigenvalues = eigenvalues

    def K(self, lengthscale, X, X2=None):
        """ Compute the mesh kernel via Laplace eigedecomposition """
        pass

    def K_diag(self, lengthscale, X):
        pass

    def Kchol(self, X, X2=None):
        pass
