"""
This module provides the :class:`MaternKarhunenLoeveKernel` kernel, the basic
kernel for discrete spectrum spaces, subclasses of :class:`DiscreteSpectrumSpace`.
"""

import logging
from math import log

import lab as B
import numpy as np
from beartype.typing import Dict, Optional

from geometric_kernels.kernels.base import BaseGeometricKernel
from geometric_kernels.lab_extras import from_numpy, is_complex
from geometric_kernels.spaces import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import Eigenfunctions


class MaternKarhunenLoeveKernel(BaseGeometricKernel):
    r"""
    This class approximates Matérn kernel by its truncated Mercer decomposition,
    in terms of the eigenfunctions & eigenvalues of the Laplacian on the space.

    .. math:: k(x, x') = \sum_{l=0}^{L-1} S(\sqrt\lambda_l) \sum_{s=1}^{d_l} f_{ls}(x) f_{ls}(x'),

    where $\lambda_l$ and $f_{ls}(\cdot)$ are the eigenvalues and
    eigenfunctions of the Laplacian such that
    $\Delta f_{ls} = \lambda_l f_{ls}$, and $S(\cdot)$ is the spectrum
    of the Matérn kernel. The eigenvalues and eigenfunctions belong to the
    :class:`~.spaces.DiscreteSpectrumSpace` instance.

    We denote

    .. math:: G_l(\cdot, \cdot') = \sum_{s=1}^{d_l} f_{ls}(\cdot) f_{ls}(\cdot')

    and term the sets $[f_{ls}]_{s=1}^{d_l}$  *"levels"*.

    For many spaces, like the sphere, we can employ addition
    theorems to efficiently compute $G_l(\cdot, \cdot')$ without calculating
    the individual $f_{ls}(\cdot)$. Note that $\lambda_l$ are not required to
    be unique: it is possible that for some $l,l'$, $\lambda_l = \lambda_{l'}$.
    In other words, the "levels" do not necessarily correspond to full
    eigenspaces. A level may even correspond to a single eigenfunction.

    .. note::
        A brief introduction into the theory behind
        :class:`MaternKarhunenLoeveKernel` can be found in
        :doc:`this </theory/compact>` & :doc:`this </theory/addition_theorem>`
        documentation pages.

    :param space:
        The space to define the kernel upon.
    :param num_levels:
        Number of levels to include in the summation.
    :param normalize:
        Whether to normalize kernel to have unit average variance. Keyword-only.
    :param log_spectrum_on:
        If True, the spectrum is computed in the log domain. Keyword-only.
    """

    def __init__(
        self,
        space: DiscreteSpectrumSpace,
        num_levels: int,
        *,
        normalize: bool = True,
        log_spectrum_on=False,
    ):
        super().__init__(space)
        self.num_levels = num_levels  # in code referred to as `L`.
        self._eigenvalues_laplacian = self.space.get_eigenvalues(self.num_levels)
        self._eigenfunctions = self.space.get_eigenfunctions(self.num_levels)
        self.normalize = normalize
        self.log_spectrum_on = log_spectrum_on

    @property
    def space(self) -> DiscreteSpectrumSpace:
        """
        The space on which the kernel is defined.
        """
        self._space: DiscreteSpectrumSpace
        return self._space

    def init_params(self) -> Dict[str, B.NPNumeric]:
        """
        Initializes the dict of the trainable parameters of the kernel.

        Returns `dict(nu=np.array([np.inf]), lengthscale=np.array([1.0]))`.

        This dict can be modified and is passed around into such methods as
        :meth:`~.K` or :meth:`~.K_diag`, as the `params` argument.

        .. note::
            The values in the returned dict are always of the NumPy array type.
            Thus, if you want to use some other backend for internal
            computations when calling :meth:`~.K` or :meth:`~.K_diag`, you
            need to replace the values with the analogs typed as arrays of
            the desired backend.
        """
        params = dict(nu=np.array([np.inf]), lengthscale=np.array([1.0]))

        return params

    def _spectrum(
        self,
        s: B.Numeric,
        nu: B.Numeric,
        lengthscale: B.Numeric,
        *,
        log_spectrum_on: Optional[bool] = None,
    ) -> B.Numeric:
        """
        The spectrum of the Matérn kernel with hyperparameters `nu` and
        `lengthscale` on the space with eigenvalues `s`.

        :param s:
            The eigenvalues of the space.
        :param nu:
            The smoothness parameter of the kernel.
        :param lengthscale:
            The length scale parameter of the kernel.
        :param log_spectrum_on:
            If True, the spectrum is computed in the log domain. Keyword-only.
            If not provided or None, the value of `self.log_spectrum_on` is used.

        :return:
            The spectrum of the Matérn kernel (or the log thereof, if
            `log_spectrum_on` is set to True).
        """
        assert lengthscale.shape == (1,)
        assert nu.shape == (1,)

        if log_spectrum_on is None:
            log_spectrum_on = self.log_spectrum_on

        # Note: 1.0 in safe_nu can be replaced by any finite positive value
        safe_nu = B.where(nu == np.inf, B.cast(B.dtype(lengthscale), np.r_[1.0]), nu)

        # for nu == np.inf
        if log_spectrum_on:
            spectral_values_nu_infinite = (
                -(lengthscale**2) / 2.0 * B.cast(B.dtype(lengthscale), s)
            )
        else:
            spectral_values_nu_infinite = B.exp(
                -(lengthscale**2) / 2.0 * B.cast(B.dtype(lengthscale), s)
            )

        # for nu < np.inf
        power = -safe_nu - self.space.dimension / 2.0
        base = 2.0 * safe_nu / lengthscale**2 + B.cast(B.dtype(safe_nu), s)
        if log_spectrum_on:
            spectral_values_nu_finite = B.log(base) * power
        else:
            spectral_values_nu_finite = base**power

        return B.where(
            nu == np.inf, spectral_values_nu_infinite, spectral_values_nu_finite
        )

    @property
    def eigenfunctions(self) -> Eigenfunctions:
        """
        Eigenfunctions of the kernel.
        """
        return self._eigenfunctions

    @property
    def eigenvalues_laplacian(self) -> B.Numeric:
        """
        Eigenvalues of the Laplacian.
        """
        return self._eigenvalues_laplacian

    def eigenvalues(
        self,
        params: Dict[str, B.Numeric],
        normalize: Optional[bool] = None,
        *,
        log_spectrum_on: Optional[bool] = None,
    ) -> B.Numeric:
        """
        Eigenvalues of the kernel.

        :param params:
            Parameters of the kernel. Must contain keys `"lengthscale"` and
            `"nu"`. The shapes of `params["lengthscale"]` and `params["nu"]`
            are `(1,)`.
        :param normalize:
            Whether to normalize kernel to have unit average variance.
            If None, uses `self.normalize` to decide.

            Defaults to None.
        :param log_spectrum_on:
            If True, the spectrum is computed in the log domain. Keyword-only.
            If not provided or None, the value of `self.log_spectrum_on` is used.

        :return:
            An [L, 1]-shaped array.
        """
        assert "lengthscale" in params
        assert params["lengthscale"].shape == (1,)
        assert "nu" in params
        assert params["nu"].shape == (1,)

        if log_spectrum_on is None:
            log_spectrum_on = self.log_spectrum_on

        spectral_values = self._spectrum(
            self.eigenvalues_laplacian,
            nu=params["nu"],
            lengthscale=params["lengthscale"],
            log_spectrum_on=log_spectrum_on,
        )

        normalize = normalize or (normalize is None and self.normalize)

        if normalize:

            if log_spectrum_on:
                if hasattr(self.eigenfunctions, "log_num_eigenfunctions_per_level"):
                    num_eigenfunctions_per_level = (
                        self.eigenfunctions.log_num_eigenfunctions_per_level
                    )
                else:
                    logging.getLogger(__name__).warning(
                        "Using log_spectrum_on on a space without"
                        " log_num_eigenfunctions_per_level."
                    )

                    num_eigenfunctions_per_level = list(
                        map(log, self.eigenfunctions.num_eigenfunctions_per_level)
                    )
            else:
                num_eigenfunctions_per_level = (
                    self.eigenfunctions.num_eigenfunctions_per_level
                )

            num_eigenfunctions_per_level = B.cast(
                B.dtype(spectral_values),
                from_numpy(spectral_values, num_eigenfunctions_per_level)[:, None],
            )

            if log_spectrum_on:
                normalizer = B.logsumexp(spectral_values + num_eigenfunctions_per_level)
                return spectral_values - normalizer
            else:
                normalizer = B.sum(spectral_values * num_eigenfunctions_per_level)
                return spectral_values / normalizer

        return spectral_values

    def K(
        self, params, X: B.Numeric, X2: Optional[B.Numeric] = None, **kwargs  # type: ignore
    ) -> B.Numeric:
        assert "lengthscale" in params
        assert params["lengthscale"].shape == (1,)
        assert "nu" in params
        assert params["nu"].shape == (1,)

        if "log_spectrum_on" in kwargs:
            log_spectrum_on = kwargs["log_spectrum_on"]
            kwargs["log_weights_on"] = log_spectrum_on
            kwargs.pop("log_spectrum_on")
        else:
            log_spectrum_on = self.log_spectrum_on
            if log_spectrum_on:
                kwargs["log_weights_on"] = True

        weights = B.cast(
            B.dtype(params["nu"]),
            self.eigenvalues(params, log_spectrum_on=log_spectrum_on),
        )  # [L, 1]
        Phi = self.eigenfunctions
        K = Phi.weighted_outerproduct(weights, X, X2, **kwargs)  # [N, N2]
        if is_complex(K):
            return B.real(K)
        else:
            return K

    def K_diag(self, params, X: B.Numeric, **kwargs) -> B.Numeric:
        assert "lengthscale" in params
        assert params["lengthscale"].shape == (1,)
        assert "nu" in params
        assert params["nu"].shape == (1,)

        if "log_spectrum_on" in kwargs:
            log_spectrum_on = kwargs["log_spectrum_on"]
            kwargs["log_weights_on"] = log_spectrum_on
            kwargs.pop("log_spectrum_on")
        else:
            log_spectrum_on = self.log_spectrum_on
            if log_spectrum_on:
                kwargs["log_weights_on"] = True

        weights = B.cast(
            B.dtype(params["nu"]),
            self.eigenvalues(params, log_spectrum_on=log_spectrum_on),
        )  # [L, 1]
        Phi = self.eigenfunctions
        K_diag = Phi.weighted_outerproduct_diag(weights, X, **kwargs)  # [N,]
        if is_complex(K_diag):
            return B.real(K_diag)
        else:
            return K_diag
