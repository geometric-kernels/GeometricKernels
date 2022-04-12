"""
Implementation of geometric kernels on several spaces
"""

import lab as B
import numpy as np

from geometric_kernels.eigenfunctions import Eigenfunctions
from geometric_kernels.kernels import BaseGeometricKernel
from geometric_kernels.lab_extras import from_numpy, logspace, trapz
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
                B.dtype(nu), from_numpy(nu, s**2)
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

        weights = B.cast(
            B.dtype(params["nu"]), self.eigenvalues(params, state)
        )  # [M, 1]
        Phi = state["eigenfunctions"]

        return Phi.weighted_outerproduct(weights, X, X2, **params)  # [N, N2]

    def K_diag(self, params, state, X: B.Numeric, **kwargs) -> B.Numeric:
        assert "eigenvalues_laplacian" in state
        assert "eigenfunctions" in state

        weights = self.eigenvalues(params, state)  # [M, 1]
        Phi = state["eigenfunctions"]

        return Phi.weighted_outerproduct_diag(weights, X, **params)  # [N,]


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
        num_points_t: int,
    ):
        r"""
        :param space: Space providing the heat kernel and distance.
        :param nu: Determines continuity of the Mat'ern kernel. Typical values include 1/2 (i.e., the Exponential kernel), 3/2, 5/2.
        :param num_point_t: number of points used in the integral.
        """

        super().__init__(space)
        self.num_points_t = num_points_t  # in code referred to as `T`.

    def init_params_and_state(self):
        """
        Get initial params and state.

        For `MaternIntegratedKernel`, params contains the lengthscale and smoothness parameter `nu`. The state is an empty `dict`.

        :return: tuple(params, state)
        """
        params = dict(lengthscale=1.0, nu=0.5)
        state = dict()

        return params, state

    def link_function(self, params, distance: B.Numeric, t: B.Numeric):
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
        assert "nu" in params
        assert "lengthscale" in params

        nu = params["nu"]
        lengthscale = params["lengthscale"]

        heat_kernel = self.space.heat_kernel(
            distance, t, self.num_points_t
        )  # (..., N1, N2, T)
        print('heat', heat_kernel.shape)
        print('t', t.shape)

        result = (
            B.power(t, nu - 1.0) * B.exp(-2.0 * nu / lengthscale**2 * t) * heat_kernel
        )

        return result

    def kernel(
        self, params, X: B.Numeric, X2: Optional[B.Numeric] = None, diag: bool = False, **kwargs  # type: ignore
    ) -> B.Numeric:
        assert "nu" in params
        assert "lengthscale" in params

        lengthscale = params["lengthscale"]

        if X2 is None:
            X2 = X

        # Compute the geodesic distance
        if diag:
            distance = self.space.distance(X, X, diag=True)
        else:
            distance = self.space.distance(X, X2, diag=False)

        shift = B.log(lengthscale) / B.log(10.0)  # Log 10
        t_vals = logspace(-2.5 + shift, 1.5 + shift, self.num_points_t)  # (T,)

        integral_vals = self.link_function(params, distance, t_vals)  # (N1, N2, T) or [N, T]

        reshape = [1] * B.rank(integral_vals)
        reshape[:-1] = B.shape(integral_vals)[:-1]  # [N1, N2, 1] or [N, 1]
        t_vals_integrator = B.tile(
            t_vals[None, :] if diag else t_vals[None, None, :],
            *reshape
        )  # (N1, N2, T) or (N, T)
        t_vals_integrator = B.cast(B.dtype(integral_vals), t_vals_integrator)  # (T, )
        print('integr, t_vals', B.dtype(integral_vals), B.dtype(t_vals_integrator))

        # Integral over heat kernel to obtain the Matérn kernel values
        kernel = trapz(integral_vals, t_vals_integrator, axis=-1)

        zero = B.cast(B.dtype(distance), from_numpy(distance, 0.0))

        integral_vals_normalizing_cst = self.link_function(params, zero, t_vals)
        t_vals_integrator = B.cast(B.dtype(integral_vals_normalizing_cst), t_vals)
        normalizing_cst = trapz(integral_vals_normalizing_cst, t_vals_integrator, axis=-1)

        return kernel / normalizing_cst

    def K(
        self, params, state, X: B.Numeric, X2: Optional[B.Numeric] = None, **kwargs  # type: ignore
    ) -> B.Numeric:
        """Compute the kernel via integration of heat kernel"""
        return self.kernel(params, X, X2, diag=False)

    def K_diag(self, params, state, X: B.Numeric, **kwargs) -> B.Numeric:
        """Compute the kernel via integration of heat kernel"""
        return self.kernel(params, X, diag=True)
