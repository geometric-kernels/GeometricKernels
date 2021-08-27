"""
Implementation of geometric kernels on several spaces
"""
from typing import Callable, Mapping, Optional

import numpy as np
import tensorflow as tf

from geometric_kernels.eigenfunctions import Eigenfunctions
from geometric_kernels.kernels import BaseGeometricKernel
from geometric_kernels.spaces.base import SpaceWithEigenDecomposition
from geometric_kernels.types import Parameter, TensorLike


class MaternKarhunenLoeveKernel(BaseGeometricKernel):
    r"""
    This class approximates a kernel by the finite feature decomposition:

    .. math:: k(x, x') = \sum_{i=0}^{M-1} S(\sqrt\lambda_i) \phi_i(x) \phi_i(x'),

    where :math:`\lambda_i` and :math:`\phi_i(\cdot)` are the eigenvalues and
    eigenfunctions of the Laplace-Beltrami operator and :math:`S(\cdot)` the
    spectrum of the stationary kernel. The eigenvalues and eigenfunctions belong
    to the `SpaceWithEigenDecomposition` instance.
    """

    def __init__(
        self,
        space: SpaceWithEigenDecomposition,
        nu: float,
        num_eigenfunctions: int,
    ):
        r"""
        :param space: Space providing the eigenvalues and eigenfunctions of
            the Laplace-Beltrami operator.
        :param nu: Determines continuity of the Mat\'ern kernel. Typical values
            include 1/2 (i.e., the Exponential kernel), 3/2, 5/2 and +\infty
            `np.inf` which corresponds to the Squared Exponential kernel.
        :param num_eigenfunctions: number of eigenvalues and functions to include
            in the summation.
        """
        super().__init__(space)
        self.nu = nu
        self.num_eigenfunctions = num_eigenfunctions  # in code referred to as `M`.

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

    def eigenfunctions(self, **__parameters) -> Eigenfunctions:
        """
        Eigenfunctions of the kernel, may depend on parameters.
        """
        return self.space.get_eigenfunctions(self.num_eigenfunctions)

    def eigenvalues(self, **parameters) -> TensorLike:
        """
        Eigenvalues of the kernel.

        :return: [M, 1]
        """
        assert "lengthscale" in parameters
        eigenvalues_laplacian = self.space.get_eigenvalues(self.num_eigenfunctions)  # [M, 1]
        return self._spectrum(eigenvalues_laplacian ** 0.5, lengthscale=parameters["lengthscale"])

    def K(
        self, X: TensorLike, X2: Optional[TensorLike] = None, **parameters: Mapping[str, Parameter]
    ) -> TensorLike:
        """Compute the mesh kernel via Laplace eigendecomposition"""
        weights = self.eigenvalues(**parameters)  # [M, 1]
        Phi = self.eigenfunctions()
        return Phi.weighted_outerproduct(weights, X, X2)  # [N, N2]

    def K_diag(self, X: TensorLike, **parameters: Mapping[str, Parameter]) -> TensorLike:
        weights = self.eigenvalues(**parameters)  # [M, 1]
        Phi = self.eigenfunctions()
        return Phi.weighted_outerproduct_diag(weights, X)  # [N,]
