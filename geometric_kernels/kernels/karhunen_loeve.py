"""
This module provides the :class:`MaternKarhunenLoeveKernel` kernel, the basic
kernel for discrete spectrum spaces, subclasses of :class:`DiscreteSpectrumSpace`.
"""

import lab as B
import numpy as np
from beartype.typing import Optional

from geometric_kernels.kernels.base import BaseGeometricKernel
from geometric_kernels.lab_extras import from_numpy, is_complex
from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import Eigenfunctions


class MaternKarhunenLoeveKernel(BaseGeometricKernel):
    r"""This class approximates a kernel by the finite feature decomposition using
    its Laplace-Beltrami eigenfunctions and eigenvalues.

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

    .. note::
        A brief introduction into the theory behind MaternKarhunenLoeveKernel
        can be found in :doc:`this </theory/compact>` and
        :doc:`this </theory/addition_theorem>` documentation pages.

    :param space: The space to define the kernel upon.
    :param num_levels: Number of levels to include in the summation.
    :param normalize: Whether to normalize kernel to have unit average variance.
    """

    def __init__(
        self,
        space: DiscreteSpectrumSpace,
        num_levels: int,
        normalize: bool = True,
    ):
        super().__init__(space)
        self.num_levels = num_levels  # in code referred to as `M`.
        self._eigenvalues_laplacian = self.space.get_eigenvalues(self.num_levels)
        self._eigenfunctions = self.space.get_eigenfunctions(self.num_levels)
        self.normalize = normalize

    def init_params(self):
        """
        :return: The initial `params` dict containing the length scale
            parameter `lengthscale` and the smoothness parameter `nu`.

        .. note::
           `nu` determines the smoothness of the Matérn kernel. Typical values
           include 1/2 (in R^n, gives the exponential kernel), 3/2, 5/2, and
           `np.inf` which corresponds to the heat kernel (in R^n, a.k.a. squared
           exponential kernel, RBF kernel, diffusion kernel, Gaussian kernel).

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
        K = Phi.weighted_outerproduct(weights, X, X2, **params)  # [N, N2]
        if is_complex(K):
            return B.real(K)
        else:
            return K

    def K_diag(self, params, X: B.Numeric, **kwargs) -> B.Numeric:
        weights = self.eigenvalues(params)  # [M, 1]
        Phi = self.eigenfunctions
        K_diag = Phi.weighted_outerproduct_diag(weights, X, **params)  # [N,]
        if is_complex(K_diag):
            return B.real(K_diag)
        else:
            return K_diag
