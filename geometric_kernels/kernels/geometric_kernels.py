"""
Implementation of geometric kernels on several spaces
"""
from typing import Callable, Mapping, Optional

import eagerpy as ep
import numpy as np
from eagerpy.tensor.tensor import Tensor

from geometric_kernels.eigenfunctions import Eigenfunctions
from geometric_kernels.kernels import BaseGeometricKernel
from geometric_kernels.spaces.base import DiscreteSpectrumSpace


class MaternKarhunenLoeveKernel(BaseGeometricKernel):
    r"""
    This class approximates a kernel by the finite feature decomposition using
    its Laplace-Beltrami eigenfunctions and eigenvalues [1, 2].

    .. math:: k(x, x') = \sum_{i=0}^{M-1} S(\sqrt\lambda_i) \phi_i(x) \phi_i(x'),

    where :math:`\lambda_i` and :math:`\phi_i(\cdot)` are the eigenvalues and
    eigenfunctions of the Laplace-Beltrami operator and :math:`S(\cdot)` the
    spectrum of the stationary kernel. The eigenvalues and eigenfunctions belong
    to the `SpaceWithEigenDecomposition` instance.

    References:

    [1] Viacheslav Borovitskiy, Alexander Terenin, Peter Mostowsky, and Marc Peter Deisenroth,
        Matern Gaussian processes on Riemannian manifolds

    [2] Arno Solin, and Simo Särkkä, Hilbert Space Methods for Reduced-Rank
        Gaussian Process Regression
    """

    def __init__(
        self,
        space: DiscreteSpectrumSpace,
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

    def _spectrum(self, s: Tensor, lengthscale: Tensor):
        """
        Matern or RBF spectrum evaluated at `s`. Depends on the
        `lengthscale` parameters.
        """
        s, lengthscale = ep.astensors(s, lengthscale)
        # cast `lengthscale` to eagerpy
        # cast `s` to the same backend as `lengthscale`
        # s = ep.from_numpy(lengthscale, s).astype(lengthscale.dtype)

        def spectrum_rbf():
            return ep.exp(-(lengthscale ** 2) / 2.0 * (s ** 2))

        def spectrum_matern():
            power = -self.nu - self.space.dimension / 2.0
            base = 2.0 * self.nu / lengthscale ** 2 + (s ** 2)
            return ep.astensor(base ** power)

        if self.nu == np.inf:
            return spectrum_rbf()
        elif self.nu > 0:
            return spectrum_matern()
        else:
            raise NotImplementedError

    def eigenfunctions(self) -> Eigenfunctions:
        """
        Eigenfunctions of the kernel, may depend on parameters.
        """
        eigenfunctions = self.space.get_eigenfunctions(self.num_eigenfunctions)
        return eigenfunctions

    def eigenvalues(self, **parameters) -> Tensor:
        """
        Eigenvalues of the kernel.

        :return: [M, 1]
        """
        assert "lengthscale" in parameters
        eigenvalues_laplacian = self.space.get_eigenvalues(self.num_eigenfunctions)  # [M, 1]
        return self._spectrum(eigenvalues_laplacian ** 0.5, lengthscale=parameters["lengthscale"])

    def K(self, X: Tensor, X2: Optional[Tensor] = None, **parameters) -> Tensor:
        """Compute the mesh kernel via Laplace eigendecomposition"""
        weights = self.eigenvalues(**parameters)  # [M, 1]
        Phi = self.eigenfunctions()
        return Phi.weighted_outerproduct(weights, X, X2, **parameters)  # [N, N2]

    def K_diag(self, X: Tensor, **parameters) -> Tensor:
        weights = self.eigenvalues(**parameters)  # [M, 1]
        Phi = self.eigenfunctions()
        return Phi.weighted_outerproduct_diag(weights, X, **parameters)  # [N,]

    # def K(self, X: TensorLike, X2: Optional[TensorLike] = None, **parameters) -> TensorLike:
    #     Phi_X = self.eigenfunctions(**parameters)(X)  # [N, L]
    #     if X2 is None:
    #         Phi_X2 = Phi_X
    #     else:
    #         Phi_X2 = self.eigenfunctions(**parameters)(X2)  # [N2, L]

    #     coeffs = self.eigenvalues(**parameters)  # [L, 1]
    #     Kxx = ep.matmul(coeffs.T * Phi_X, Phi_X2.T)  # [N, N2]
    #     return Kxx.raw

    # def K_diag(self, X, **parameters):
    #     Phi_X = self.eigenfunctions(**parameters)(X)  # [N, L]
    #     coeffs = self.eigenvalues(**parameters)  # [L, 1]
    #     Kx = ep.sum(coeffs.T * Phi_X ** 2, axis=1)  # [N,]
    #     return Kx.raw
