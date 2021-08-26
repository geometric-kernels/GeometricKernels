"""
Implementation of geometric kernels on several spaces
"""
from typing import Callable, Optional

import eagerpy as ep
import numpy as np

from geometric_kernels.kernels import BaseGeometricKernel
from geometric_kernels.spaces import Mesh
from geometric_kernels.types import Parameter, TensorLike


class MeshKernel(BaseGeometricKernel):
    """
    Geometric kernel on a Mesh
    """

    def __init__(
        self,
        space: Mesh,
        nu: float,
        truncation_level: int,
    ):
        super().__init__(space)
        self.truncation_level = truncation_level
        self.nu = nu
        self._eigenfunctions: Optional[Callable[[TensorLike], TensorLike]] = None

    def spectrum(self, s: TensorLike, lengthscale: Parameter):
        """
        Matern or RBF spectrum evaluated as `s`. Depends on the
        `lengthscale` parameters.
        """

        def spectrum_rbf():
            return ep.exp(-(lengthscale ** 2) / 2.0 * (s ** 2))

        def spectrum_matern():
            power = -self.nu - self.space.dim / 2.0
            base = 2.0 * self.nu / lengthscale ** 2 + (s ** 2)
            return ep.astensor(base ** power)

        if self.nu == np.inf:
            return spectrum_rbf()
        elif self.nu > 0:
            return spectrum_matern()
        else:
            raise NotImplementedError

    def eigenfunctions(self, **__parameters) -> Callable:
        """
        Eigenfunctions of the kernel, may depend on parameters.
        """

        class _EigenFunctions:
            """
            Converts the array of eigenvectors to a callable objects,
            The inputs are given by the indices.
            """

            def __init__(self, eigenvectors):
                self.eigenvectors = ep.astensor(eigenvectors)

            def __call__(self, indices: TensorLike) -> TensorLike:
                """
                Selects N locations from the  eigenvectors.

                :param indices: indices [N, 1]
                :return: [N, L]
                """
                assert len(indices.shape) == 2
                assert indices.shape[-1] == 1
                indices = ep.from_numpy(self.eigenvectors, indices).astype(np.int32)  # [I, 1]

                # This is a very hacky way of taking along 0'th axis.
                # For some reason eagerpy does not take along axis other than last.
                Phi = self.eigenvectors.T.take_along_axis(indices.T, axis=-1).T
                return Phi

        if self._eigenfunctions is None:
            eigenvectors = self.space.get_eigenfunctions(self.truncation_level)  # [Nv, L]
            self._eigenfunctions = _EigenFunctions(eigenvectors)

        return self._eigenfunctions

    def eigenvalues(self, **parameters) -> TensorLike:
        """
        Eigenvalues of the kernel.

        :return: [L, 1]
        """
        assert "lengthscale" in parameters
        eigenvalues_laplacian = self.space.get_eigenvalues(self.truncation_level)  # [L, 1]
        return self.spectrum(eigenvalues_laplacian ** 0.5, lengthscale=parameters["lengthscale"])

    def K(self, X, X2=None, **parameters):
        """Compute the mesh kernel via Laplace eigendecomposition"""
        Phi_X = self.eigenfunctions()(X)  # [N, L]
        if X2 is None:
            Phi_X2 = Phi_X
        else:
            Phi_X2 = self.eigenfunctions()(X2)  # [N2, L]

        coeffs = self.eigenvalues(**parameters)  # [L, 1]
        Kxx = ep.matmul(coeffs.T * Phi_X, Phi_X2.T)
        return Kxx.raw

    def K_diag(self, X, **parameters):
        Phi_X = self.eigenfunctions()(X)  # [N, L]
        coeffs = self.eigenvalues(**parameters)  # [L, 1]
        Kx = ep.sum(coeffs.T * Phi_X ** 2, axis=1)  # [N, ]
        return Kx.raw
