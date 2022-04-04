"""
Implementation of geometric kernels on several spaces
"""

import lab as B
import numpy as np

from geometric_kernels.eigenfunctions import Eigenfunctions
from geometric_kernels.kernels import BaseGeometricKernel
from geometric_kernels.lab_extras import from_numpy
from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.utils import Optional


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
        self.num_eigenfunctions = num_eigenfunctions  # in code referred to as `M`.

    def init_params_and_state(self):
        """
        Get initial params and state.

        In this case, state is Laplacian eigenvalues and eigenfunctions,
        and params contains the lengthscale and smoothness parameter `nu`.

        :return: tuple(params, state)
        """
        params = dict(lengthscale=1.0, nu=0.5)

        eigenvalues_laplacian = self.space.get_eigenvalues(self.num_eigenfunctions)
        eigenfunctions = self.space.get_eigenfunctions(self.num_eigenfunctions)

        state = dict(
            eigenvalues_laplacian=eigenvalues_laplacian, eigenfunctions=eigenfunctions
        )

        return params, state

    def _spectrum(
        self, s: B.Numeric, nu: B.Numeric, lengthscale: B.Numeric
    ) -> B.Numeric:
        """
        Matern or RBF spectrum evaluated at `s`.
        Depends on the `lengthscale` parameters.
        """
        if nu == np.inf:
            return B.exp(-(lengthscale**2) / 2.0 * (s**2))
        elif nu > 0:
            power = -nu - self.space.dimension / 2.0
            base = 2.0 * nu / lengthscale**2 + B.cast(
                B.dtype(nu),
                from_numpy(nu, s**2)
            )
            return base**power
        else:
            raise NotImplementedError

    def eigenfunctions(self) -> Eigenfunctions:
        """
        Eigenfunctions of the kernel, may depend on parameters.
        """
        eigenfunctions = self.space.get_eigenfunctions(self.num_eigenfunctions)
        return eigenfunctions

    def eigenvalues(self, params, state) -> B.Numeric:
        """
        Eigenvalues of the kernel.

        :return: [M, 1]
        """
        assert "lengthscale" in params
        assert "nu" in params

        assert "eigenvalues_laplacian" in state

        eigenvalues_laplacian = state["eigenvalues_laplacian"]  # [M, 1]
        return self._spectrum(
            eigenvalues_laplacian**0.5,
            nu=params["nu"],
            lengthscale=params["lengthscale"],
        )

    def K(
        self, params, state, X: B.Numeric, X2: Optional[B.Numeric] = None, **kwargs  # type: ignore
    ) -> B.Numeric:
        """Compute the mesh kernel via Laplace eigendecomposition"""
        assert "eigenfunctions" in state
        assert "eigenvalues_laplacian" in state

        weights = B.cast(B.dtype(params["nu"]), self.eigenvalues(params, state))  # [M, 1]
        Phi = state["eigenfunctions"]

        return Phi.weighted_outerproduct(weights, X, X2, **params)  # [N, N2]

    def K_diag(self, params, state, X: B.Numeric, **kwargs) -> B.Numeric:
        assert "eigenvalues_laplacian" in state
        assert "eigenfunctions" in state

        weights = self.eigenvalues(params, state)  # [M, 1]
        Phi = state["eigenfunctions"]

        return Phi.weighted_outerproduct_diag(weights, X, **params)  # [N,]
