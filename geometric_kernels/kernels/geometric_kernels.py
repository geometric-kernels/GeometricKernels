"""
Implementation of geometric kernels on several spaces
"""
from typing import Callable, Optional

import numpy as np
import tensorflow as tf

from geometric_kernels.kernels import BaseGeometricKernel
from geometric_kernels.spaces.base import SpaceWithEigenDecomposition
from geometric_kernels.types import Parameter, TensorLike


class MaternKarhunenLoeveKernel(BaseGeometricKernel):
    r"""
    This class approximates a kernel by the finite feature decomposition:

    .. math:: k(x, x') = \sum_{i=0}^L S(\sqrt\lambda_i) \phi_i(x) \phi_i(x'),

    where :math:`\lambda_i` and :math:`\phi_i(\cdot)` are the eigenvalues and
    eigenfunctions of the Laplace-Beltrami operator and :math:`S(\cdot)` the
    spectrum of the stationary kernel. The eigenvalues and eigenfunctions belong
    to the `SpaceWithEigenDecomposition` instance.
    """

    def __init__(
        self,
        space: SpaceWithEigenDecomposition,
        nu: float,
        num_components: int,
    ):
        r"""
        :param space: Space providing the eigenvalues and eigenfunctions of
            the Laplace-Beltrami operator.
        :param nu: Determines continuity of the Mat\'ern kernel. Typical values
            include 1/2 (i.e., the Exponential kernel), 3/2, 5/2 and +\infty
            `np.inf` which corresponds to the Squared Exponential kernel.
        :param num_components: number of eigenvalues and functions to include
            in the summation.
        """
        super().__init__(space)
        self.nu = nu
        self.num_components = num_components  # in code referred to as `L`.

    def _spectrum(self, s: TensorLike, lengthscale: Parameter):
        """
        Matern or RBF spectrum evaluated at `s`. Depends on the
        `lengthscale` parameters.
        """

        def spectrum_rbf():
            return tf.exp(-(lengthscale ** 2) / 2.0 * (s ** 2))

        def spectrum_matern():
            power = -self.nu - self.space.dimension / 2.0
            base = 2.0 * self.nu / lengthscale ** 2 + (s ** 2)
            return base ** power

        if self.nu == np.inf:
            return spectrum_rbf()
        elif self.nu > 0:
            return spectrum_matern()
        else:
            raise NotImplementedError

    def eigenfunctions(self, **__parameters) -> Callable[[TensorLike], TensorLike]:
        """
        Eigenfunctions of the kernel, may depend on parameters.
        """
        return self.space.get_eigenfunctions(self.num_components)

    def eigenvalues(self, **parameters) -> TensorLike:
        """
        Eigenvalues of the kernel.

        :return: [L, 1]
        """
        assert "lengthscale" in parameters
        eigenvalues_laplacian = self.space.get_eigenvalues(self.num_components)  # [L, 1]
        return self._spectrum(eigenvalues_laplacian ** 0.5, lengthscale=parameters["lengthscale"])

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
