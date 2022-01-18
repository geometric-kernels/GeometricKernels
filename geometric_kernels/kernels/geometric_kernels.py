"""
Implementation of geometric kernels on several spaces
"""

import lab as B
import numpy as np

from geometric_kernels.eigenfunctions import Eigenfunctions
from geometric_kernels.kernels import BaseGeometricKernel
from geometric_kernels.lab_extras import logspace, trapz
from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.hyperbolic import Hyperbolic
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

    def _spectrum(self, s: B.Numeric, lengthscale: B.Numeric) -> B.Numeric:
        """
        Matern or RBF spectrum evaluated at `s`. Depends on the
        `lengthscale` parameters.
        """
        if self.nu == np.inf:
            return B.exp(-(lengthscale ** 2) / 2.0 * (s ** 2))
        elif self.nu > 0:
            power = -self.nu - self.space.dimension / 2.0
            base = 2.0 * self.nu / lengthscale ** 2 + (s ** 2)
            return base ** power
        else:
            raise NotImplementedError

    def eigenfunctions(self) -> Eigenfunctions:
        """
        Eigenfunctions of the kernel, may depend on parameters.
        """
        eigenfunctions = self.space.get_eigenfunctions(self.num_eigenfunctions)
        return eigenfunctions

    def eigenvalues(self, **parameters) -> B.Numeric:
        """
        Eigenvalues of the kernel.

        :return: [M, 1]
        """
        assert "lengthscale" in parameters
        eigenvalues_laplacian = self.space.get_eigenvalues(
            self.num_eigenfunctions
        )  # [M, 1]
        return self._spectrum(
            eigenvalues_laplacian ** 0.5,
            lengthscale=parameters["lengthscale"],
        )

    def K(
        self, X: B.Numeric, X2: Optional[B.Numeric] = None, **parameters  # type: ignore
    ) -> B.Numeric:
        """Compute the mesh kernel via Laplace eigendecomposition"""
        weights = self.eigenvalues(**parameters)  # [M, 1]
        Phi = self.eigenfunctions()
        return Phi.weighted_outerproduct(weights, X, X2, **parameters)  # [N, N2]

    def K_diag(self, X: B.Numeric, **parameters) -> B.Numeric:
        weights = self.eigenvalues(**parameters)  # [M, 1]
        Phi = self.eigenfunctions()
        return Phi.weighted_outerproduct_diag(weights, X, **parameters)  # [N,]


class MaternIntegratedKernel(BaseGeometricKernel):
    r"""
    This class computes a Matérn kernel by integrating over the heat kernel [1].

    For non-compact manifolds:
    .. math:: k_{\nu, \kappa, \sigma^2}(x, x') = \int_0^{\infty} u^{\nu - 1} e^{-\frac{2 \nu}{\kappa^2} u} k_{\infty, \sqrt{2 u}, \sigma^2}(x, x') \d u

    For compact manifolds:
    .. math:: k_{\nu, \kappa, \sigma^2}(x, x') = \int_0^{\infty} u^{\nu - 1 + d/2} e^{-\frac{2 \nu}{\kappa^2} u} k_{\infty, \sqrt{2 u}, \sigma^2}(x, x') \d u

    References:

    [1] N. Jaquier, V. Borovitskiy, A. Smolensky, A. Terenin, T. Afour, and L. Rozo.
        Geometry-aware Bayesian Optimization in Robotics using Riemannian Matérn Kernels. CoRL 2021.
    """

    def __init__(
        self,
        space: Hyperbolic,
        nu: float,
        num_points_t: int,
    ):
        r"""
        :param space: Space providing the heat kernel and distance.
        :param nu: Determines continuity of the Mat'ern kernel. Typical values include 1/2 (i.e., the Exponential kernel), 3/2, 5/2.
        :param num_point_t: number of points used in the integral.
        """

        super().__init__(space)
        self.nu = nu
        self.num_points_t = num_points_t  # in code referred to as `T`.

    def link_function(self, distance: B.Numeric, t: B.Numeric, lengthscale: B.Numeric, **parameters):
        r"""
        This function links the heat kernel to the Matérn kernel, i.e., the Matérn kernel correspond to the integral of
        this function from 0 to inf.
        Parameters
        ----------
        :param distance: precomputed distance between the inputs
        :param t: heat kernel lengthscale
        :param lenghtscale: lengthscale parameter of the kernel
        Returns
        -------
        :return: link function between the heat and Matérn kernels
        """
        heat_kernel = self.space.heat_kernel(
            distance, t, **parameters
        )  # (..., N1, N2, T)

        if self.space.is_compact:
            power_term = B.power(t, self.nu + self.space.dimension/2 - 1.0)
        else:
            power_term = B.power(t, self.nu - 1.0)

        result = (
            power_term
            * B.exp(-2.0 * self.nu / lengthscale ** 2 * t)
            * heat_kernel
        )

        return result

    def K(
        self, X: B.Numeric, X2: Optional[B.Numeric] = None, **parameters  # type: ignore
    ) -> B.Numeric:
        """Compute the kernel via integration of heat kernel"""
        assert "lengthscale" in parameters
        lengthscale = parameters["lengthscale"]
        parameters.pop("lengthscale")

        # Compute the geodesic distance
        distance = self.space.distance(X, X2, diag=False)

        shift = B.log(lengthscale) / B.log(10.0)  # Log 10
        t_vals = logspace(-2.5 + shift, 1.5 + shift, self.num_points_t)  # (T,)

        integral_vals = self.link_function(distance, t_vals, lengthscale, **parameters)

        # Integral over heat kernel to obtain the Matérn kernel values
        kernel = trapz(integral_vals, t_vals, axis=-1)

        integral_vals_normalizing_cst = self.link_function(
            0.0, t_vals, lengthscale=lengthscale
        )
        normalizing_cst = trapz(integral_vals_normalizing_cst, t_vals, axis=-1)

        return kernel / normalizing_cst

    def K_diag(self, X: B.Numeric, **parameters) -> B.Numeric:
        raise NotImplementedError
