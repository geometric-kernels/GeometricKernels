"""
Implementation of geometric kernels on several spaces
"""
from typing import Callable, Optional

import numpy as np
import tensorflow as tf

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
            return tf.exp(-(lengthscale ** 2) / 2.0 * (s ** 2))

        def spectrum_matern():
            power = -self.nu - self.space.dim / 2.0
            base = 2.0 * self.nu / lengthscale ** 2 + (s ** 2)
            return base ** power

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
                self.eigenvectors = eigenvectors

            def __call__(self, indices: TensorLike) -> TensorLike:
                """
                Selects N locations from the  eigenvectors.

                :param indices: indices [N, 1]
                :return: [N, L]
                """
                assert len(indices.shape) == 2
                assert indices.shape[-1] == 1
                indices = tf.cast(indices, dtype=tf.int32)
                Phi = tf.gather(self.eigenvectors, tf.reshape(indices, (-1,)), axis=0)
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
        Kxx = tf.matmul(Phi_X, tf.transpose(coeffs) * Phi_X2, transpose_b=True)  # [N, N2]
        return Kxx

    def K_diag(self, X, **parameters):
        Phi_X = self.eigenfunctions()(X)  # [N, L]
        coeffs = self.eigenvalues(**parameters)  # [L, 1]
        Kx = tf.reduce_sum(Phi_X ** 2 * tf.transpose(coeffs), axis=1)  # [N,]
        return Kx


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
