"""
Implementation of geometric kernels on several spaces
"""

import lab as B
import numpy as np

from geometric_kernels.eigenfunctions import Eigenfunctions
from geometric_kernels.kernels import BaseGeometricKernel
from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.hyperbolic import Hyperbolic
from geometric_kernels.utils import Optional
from geometric_kernels.extras import trapz


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
    def __init__(
            self,
            space: Hyperbolic,
            nu: float,
            num_points_t: int,
            num_points_b: int,
    ):

        super().__init__(space)
        self.nu = nu
        self.num_points_t = num_points_t  # in code referred to as `T`.
        self.num_points_b = num_points_b

    def K(
        self, X: B.Numeric, X2: Optional[B.Numeric] = None, **parameters  # type: ignore
    ) -> B.Numeric:
        """Compute the kernel via integration of heat kernel"""
        assert "lengthscale" in parameters
        lengthscale = parameters["lengthscale"]

        # Compute cosh of hyperbolic distance
        cosh_distance = B.cosh(self.space.distance(X1, X2, diag=False))

        # Evaluate integral: this is a double integral, one for computing the heat kernel, and one for the Matérn
        shift = B.log(lengthscale)  # Log 10
        t_vals = B.logspace(-5 + shift, 3 + shift, self.num_points_t)  # TODO
        b_vals = B.logspace(-4, 1.5, self.num_points_b)  # TODO

        # self.link_function(cosh_distance, t_vals[0], b_vals[0])

        # integral_vals = torch.zeros([self.nb_points_integral_t, self.nb_points_integral_b] + list(cosh_distance.shape))
        # for i in range(self.nb_points_integral_t):
        #     for j in range(self.nb_points_integral_b):
        #         integral_vals[i, j] = self.link_function(cosh_distance, t_vals[i], b_vals[j])
        # build a grid
        tt, bb = None, None

        integral_vals = self.space.link_function(cosh_distance, tt, bb, self.nu, lengthscale)

        # Integral to obtain the heat kernel values
        heat_kernel_integral_vals = trapz(integral_vals, b_vals, axis=1)
        # Integral over heat kernel to obtain the Matérn kernel values
        kernel = trapz(heat_kernel_integral_vals, t_vals, axis=0)

        # heat_kernel_integral_vals = torch.trapz(integral_vals, b_vals, dim=1)
        # # Integral over heat kernel to obtain the Matérn kernel values
        # kernel = torch.trapz(heat_kernel_integral_vals, t_vals, dim=0)

        # Evaluate the integral for the normalizing constant
        # integral_vals_normalizing_cst = torch.zeros(self.nb_points_integral_t, self.nb_points_integral_b)
        # for i in range(self.nb_points_integral_t):
        #     for j in range(self.nb_points_integral_b):
        #         integral_vals_normalizing_cst[i, j] = self.link_function(torch.ones(1, 1), t_vals[i], b_vals[j])
        # Normalizing constant
        # normalizating_cst = torch.trapz(torch.trapz(integral_vals_normalizing_cst, b_vals, dim=1), t_vals, dim=0)

        integral_vals_normalizing_cst = self.link_function(1.0, tt, bb)
        normalizing_cst = trapz(trapz(integral_vals_normalizing_cst, b_vals, axis=1), t_vals, axis=0)

        return kernel / normalizing_cst

    def K_diag(self, X: B.Numeric, **parameters) -> B.Numeric:
        pass
