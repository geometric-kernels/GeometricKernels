"""
Implementation of geometric kernels on several spaces
"""

import lab as B
import numpy as np
from opt_einsum import contract as einsum

from geometric_kernels.kernels.base import BaseGeometricKernel
from geometric_kernels.lab_extras import from_numpy, logspace, trapz
from geometric_kernels.spaces.base import DiscreteSpectrumSpace, Space
from geometric_kernels.spaces.eigenfunctions import Eigenfunctions
from geometric_kernels.spaces.hyperbolic import Hyperbolic
from geometric_kernels.utils.utils import Optional, make_deterministic


class MaternKarhunenLoeveKernel(BaseGeometricKernel):
    r"""This class approximates a kernel by the finite feature decomposition using
    its Laplace-Beltrami eigenfunctions and eigenvalues [1, 2].

    .. math:: k(x, x') = \sum_{i=0}^{M-1} S(\sqrt\lambda_i) \sum_{j=0}^{K_i} \phi_{ij}(x) \phi_{ij}(x'),

    where :math:`\lambda_i` and :math:`\phi_{ij}(\cdot)` are the
    eigenvalues and eigenfunctions of the Laplace-Beltrami operator
    such that :math:`\Delta \phi_{ij} = \lambda_i \phi_{ij}`, and
    :math:`S(\cdot)` is the spectrum of the stationary kernel. The
    eigenvalues and eigenfunctions belong to the
    `DiscreteSpectrumSpace` instance.

    We refer to the pairs :math:`(\lambda_i, G_i(\cdot, \cdot'))`
    where :math:`G_i(\cdot, \cdot') = \sum_{j=0}^{K_i}
    \phi_{ij}(\cdot) \phi_{ij}(\cdot')` as "levels". For many spaces,
    like the sphere, we can employ addition theorems to efficiently
    compute :math:`G_i(\cdot, \cdot')` without calculating the
    individual :math:`\phi_{ij}`. Note that :math:`\lambda_i` are not
    required to be unique: it is possible that for some :math:`i,j`,
    :math:`\lambda_i = \lambda_j`. In other words, the "levels" do not
    necessarily correspond to full eigenspaces. A level may even correspond
    to a single eigenfunction.

    References:

    [1] Viacheslav Borovitskiy, Alexander Terenin, Peter Mostowsky, and Marc Peter Deisenroth,
        Matern Gaussian processes on Riemannian manifolds

    [2] Arno Solin, and Simo Särkkä, Hilbert Space Methods for Reduced-Rank
        Gaussian Process Regression

    """

    def __init__(
        self,
        space: DiscreteSpectrumSpace,
        num_levels: int,
        normalize: bool = True,
    ):
        r"""
        :param space: Space providing the eigenvalues and eigenfunctions of
            the Laplace-Beltrami operator.
        :param nu: Determines continuity of the Mat\'ern kernel. Typical values
            include 1/2 (i.e., the Exponential kernel), 3/2, 5/2 and +\infty
            `np.inf` which corresponds to the Squared Exponential kernel.
        :param num_levels: number of levels to include in the summation.
        :param normalize: whether to normalize to have unit average variance.
        """
        super().__init__(space)
        self.num_levels = num_levels  # in code referred to as `M`.
        self._eigenvalues_laplacian = self.space.get_eigenvalues(self.num_levels)
        self._eigenfunctions = self.space.get_eigenfunctions(self.num_levels)
        self.normalize = normalize

    def init_params(self):
        """
        Get initial params.

        params contains the lengthscale and the smoothness parameter `nu`.

        :return: params
        """
        params = dict(lengthscale=np.array(1.0), nu=np.array(np.inf))

        return params

    def _spectrum(
        self, s: B.Numeric, nu: B.Numeric, lengthscale: B.Numeric
    ) -> B.Numeric:
        """
        Matern or RBF spectrum evaluated at `s`.
        Depends on the `lengthscale` parameters.
        """

        # Note: 1.0 in safe_nu can be replaced by any finite positive value
        safe_nu = B.where(nu == np.inf, B.cast(B.dtype(lengthscale), np.r_[1.0]), nu)

        # for nu == np.inf
        spectral_values_nu_infinite = B.exp(
            -(lengthscale**2) / 2.0 * B.cast(B.dtype(lengthscale), s**2)
        )

        # for nu < np.inf
        power = -safe_nu - self.space.dimension / 2.0
        base = 2.0 * safe_nu / lengthscale**2 + B.cast(B.dtype(safe_nu), s**2)
        spectral_values_nu_finite = base**power

        return B.where(
            nu == np.inf, spectral_values_nu_infinite, spectral_values_nu_finite
        )

    @property
    def eigenfunctions(self) -> Eigenfunctions:
        """
        Eigenfunctions of the kernel, may depend on parameters.
        """
        return self._eigenfunctions

    @property
    def eigenvalues_laplacian(self) -> B.Numeric:
        """
        Eigenvalues of the Laplacian.
        """
        return self._eigenvalues_laplacian

    def eigenvalues(self, params, normalize: Optional[bool] = None) -> B.Numeric:
        """
        Eigenvalues of the kernel.

        :return: [M, 1]
        """
        assert "lengthscale" in params
        assert "nu" in params

        spectral_values = self._spectrum(
            self.eigenvalues_laplacian**0.5,
            nu=params["nu"],
            lengthscale=params["lengthscale"],
        )
        normalize = normalize or (normalize is None and self.normalize)
        if normalize:
            normalizer = B.sum(
                spectral_values
                * B.cast(
                    B.dtype(spectral_values),
                    from_numpy(
                        spectral_values,
                        self.eigenfunctions.num_eigenfunctions_per_level,
                    )[:, None],
                )
            )
            return spectral_values / normalizer
        return spectral_values

    def K(
        self, params, X: B.Numeric, X2: Optional[B.Numeric] = None, **kwargs  # type: ignore
    ) -> B.Numeric:
        """Compute the mesh kernel via Laplace eigendecomposition"""
        weights = B.cast(B.dtype(params["nu"]), self.eigenvalues(params))  # [M, 1]
        Phi = self.eigenfunctions

        return Phi.weighted_outerproduct(weights, X, X2, **params)  # [N, N2]

    def K_diag(self, params, X: B.Numeric, **kwargs) -> B.Numeric:
        weights = self.eigenvalues(params)  # [M, 1]
        Phi = self.eigenfunctions

        return Phi.weighted_outerproduct_diag(weights, X, **params)  # [N,]


class MaternFeatureMapKernel(BaseGeometricKernel):
    r"""
    This class computes a (Matérn) kernel based on a feature map.

    For every kernel `k` on a space `X`, there is a map :math:`\phi` from the space `X`
    to some (possibly infinite-dimensional) space :math:`\mathcal{H}` such that:

    .. math :: k(x, y) = \langle \phi(x), \phi(y) \rangle_{\mathcal{H}}

    where :math:`\langle \cdot , \rangle_{\mathcal{H}}` means inner product.

    One can approximate the kernel using a finite-dimensional approximation to
    :math:`\phi` which we call a `feature map`.

    What makes this kernel specifically Matérn is that it has
    a smoothness parameter `nu` and a lengthscale parameter `lengthscale`.
    """

    def __init__(self, space: Space, feature_map, key, normalize=True):
        super().__init__(space)
        self.feature_map = make_deterministic(feature_map, key)
        self.normalize = normalize

    def init_params(self):
        params = dict(nu=np.array(np.inf), lengthscale=np.array(1.0))
        return params

    def K(self, params, X, X2=None, **kwargs):
        _, features_X = self.feature_map(
            X, params, normalize=self.normalize, **kwargs
        )  # [N, O]
        if X2 is not None:
            _, features_X2 = self.feature_map(
                X2, params, normalize=self.normalize, **kwargs
            )  # [M, O]
        else:
            features_X2 = features_X

        feature_product = einsum("...no,...mo->...nm", features_X, features_X2)
        return feature_product

    def K_diag(self, params, X, **kwargs):
        _, features_X = self.feature_map(
            X, params, normalize=self.normalize, **kwargs
        )  # [N, O]
        return B.sum(features_X**2, axis=-1)  # [N, ]


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
        :param num_point_t: number of points used in the integral.
        """

        super().__init__(space)
        self.num_points_t = num_points_t  # in code referred to as `T`.

    def init_params(self):
        """
        Get initial params.

        For `MaternIntegratedKernel`, params contains the lengthscale and smoothness parameter `nu`.

        :return: params
        """
        params = dict(lengthscale=1.0, nu=np.inf)
        return params

    def link_function(self, params, distance: B.Numeric, t: B.Numeric):
        r"""
        This function links the heat kernel to the Matérn kernel, i.e., the Matérn kernel correspond to the integral of
        this function from 0 to inf.
        Parameters
        ----------
        :param distance: precomputed distance between the inputs
        :param params: dictionary with `lengthscale` - the kernel lengthscale and `nu` - the smoothness parameter
        :param t: the heat kernel lengthscales to integrate against
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
        t_vals = B.reshape(
            logspace(-2.5 + shift, 1.5 + shift, self.num_points_t), -1
        )  # (T,)

        integral_vals = self.link_function(
            params, distance, t_vals
        )  # (N1, N2, T) or (N, T)

        reshape = [1] * B.rank(integral_vals)
        reshape[:-1] = B.shape(integral_vals)[:-1]  # (N1, N2, 1) or (N, 1)
        t_vals_integrator = B.tile(
            t_vals[None, :] if diag else t_vals[None, None, :], *reshape
        )  # (N1, N2, T) or (N, T)
        t_vals_integrator = B.cast(
            B.dtype(integral_vals), t_vals_integrator
        )  # (N1, N2, T) or (N, T)

        # Integral over heat kernel to obtain the Matérn kernel values
        kernel = trapz(integral_vals, t_vals_integrator, axis=-1)

        zero = B.cast(B.dtype(distance), from_numpy(distance, np.array(0.0)))

        integral_vals_normalizing_cst = self.link_function(params, zero, t_vals)
        t_vals_integrator = B.cast(B.dtype(integral_vals_normalizing_cst), t_vals)
        normalizing_cst = trapz(
            integral_vals_normalizing_cst, t_vals_integrator, axis=-1
        )

        return kernel / normalizing_cst

    def K(
        self, params, X: B.Numeric, X2: Optional[B.Numeric] = None, **kwargs  # type: ignore
    ) -> B.Numeric:
        """Compute the kernel via integration of heat kernel"""
        return self.kernel(params, X, X2, diag=False)

    def K_diag(self, params, X: B.Numeric, **kwargs) -> B.Numeric:
        """Compute the kernel via integration of heat kernel"""
        return self.kernel(params, X, diag=True)
