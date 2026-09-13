r"""
This module provides the :class:`MaternKernelHammingGraph` kernel, a subclass of
:class:`MaternKarhunenLoeveKernel` for :class:`HammingGraph` and
:class:`HypercubeGraph` spaces with log-domain spectral weighting and a
closed-form heat kernel when $\nu = \infty$.
"""

from math import log

import lab as B
import numpy as np
from beartype.typing import Dict, Optional, Union

from geometric_kernels.kernels.karhunen_loeve import MaternKarhunenLoeveKernel
from geometric_kernels.lab_extras import from_numpy
from geometric_kernels.spaces.eigenfunctions import Eigenfunctions
from geometric_kernels.spaces.hamming_graph import HammingGraph
from geometric_kernels.spaces.hamming_graph_eigenfunctions import (
    HammingGraphEigenfunctions,
)
from geometric_kernels.spaces.hypercube_graph import HypercubeGraph
from geometric_kernels.utils.kernel_formulas.hamming_graph import (
    _log_hamming_graph_heat_kernel,
)
from geometric_kernels.utils.utils import _check_1_vector, _check_field_in_params


class MaternKernelHammingGraph(MaternKarhunenLoeveKernel):
    r"""
    For $\nu = \infty$, there exists a closed-form formula for the heat kernel
    on hamming graphs :class:`HammingGraph` (including the binary hypercube case
    :class:`HypercubeGraph`). This class extends :class:`MaternKarhunenLoeveKernel`
    to implement this formula in the case of $\nu = \infty$ for efficiency.

    .. note::
        We only use the closed form expression if `num_levels` is `d + 1` which
        corresponds to exact computation. When truncated to fewer levels, we
        must use the parent class implementation to ensure consistency with
        feature map approximations.
    """

    def __init__(
        self,
        space: Union[HammingGraph, HypercubeGraph],
        num_levels: int,
        normalize: bool = True,
        eigenvalues_laplacian: Optional[B.Numeric] = None,
        eigenfunctions: Optional[Eigenfunctions] = None,
    ):
        if not isinstance(space, (HammingGraph, HypercubeGraph)):
            raise ValueError(
                f"`space` must be an instance of HammingGraph or HypercubeGraph, but got {type(space)}"
            )

        super().__init__(
            space,
            num_levels,
            normalize,
            eigenvalues_laplacian,
            eigenfunctions,
        )

    @staticmethod
    def log_spectrum(
        s: B.Numeric, nu: B.Numeric, lengthscale: B.Numeric, dimension: int
    ) -> B.Numeric:
        """Log Matérn spectrum, computed without forming spectral values.

        ``s`` contains Laplacian eigenvalues. The result has the same shape;
        ``nu`` and ``lengthscale`` must have shape [1].
        """
        _check_1_vector(lengthscale, "lengthscale")
        _check_1_vector(nu, "nu")
        s = B.cast(B.dtype(lengthscale), s)
        safe_nu = B.where(nu == np.inf, B.ones(lengthscale), nu)
        safe_lengthscale = B.where(nu == np.inf, B.ones(lengthscale), lengthscale)
        finite = -(safe_nu + dimension / 2.0) * B.log(
            2.0 * safe_nu / safe_lengthscale**2 + s
        )
        infinite = -(lengthscale**2) * s / 2.0
        return B.where(nu == np.inf, infinite, finite)

    @staticmethod
    def spectrum(
        s: B.Numeric, nu: B.Numeric, lengthscale: B.Numeric, dimension: int
    ) -> B.Numeric:
        """Matérn spectrum; very small linear values may underflow."""
        return B.exp(
            MaternKernelHammingGraph.log_spectrum(s, nu, lengthscale, dimension)
        )

    def _log_weights(self, params):
        _check_field_in_params(params, "lengthscale")
        _check_field_in_params(params, "nu")
        log_spectrum = self.log_spectrum(
            self.eigenvalues_laplacian,
            params["nu"],
            params["lengthscale"],
            self.space.dimension,
        )
        if isinstance(self.eigenfunctions, HammingGraphEigenfunctions):
            log_multiplicities = self.eigenfunctions._log_multiplicities(log_spectrum)
        else:
            # Preserve support for explicitly supplied generic eigenfunctions.
            log_multiplicities = B.cast(
                B.dtype(log_spectrum),
                from_numpy(
                    log_spectrum,
                    np.array(
                        [
                            log(n)
                            for n in self.eigenfunctions.num_eigenfunctions_per_level
                        ]
                    ),
                ),
            )[:, None]
        log_levels = log_spectrum + log_multiplicities
        return log_spectrum, log_levels, B.logsumexp(log_levels)

    def log_eigenvalues(self, params: Dict[str, B.Numeric]) -> B.Numeric:
        """Per-eigenfunction log kernel eigenvalues, shape [L, 1].

        With normalization enabled, normalize over the selected levels,
        accounting for their multiplicities. Otherwise return the raw log spectrum.
        """
        log_spectrum, _, log_normalizer = self._log_weights(params)
        return log_spectrum - log_normalizer if self.normalize else log_spectrum

    def eigenvalues(self, params: Dict[str, B.Numeric]) -> B.Numeric:
        """Per-eigenfunction eigenvalues; linear outputs may underflow."""
        return B.exp(self.log_eigenvalues(params))

    def K(
        self,
        params: Dict[str, B.Numeric],
        X: B.Numeric,
        X2: Optional[B.Numeric] = None,
        **kwargs,
    ) -> B.Numeric:
        if not isinstance(self.eigenfunctions, HammingGraphEigenfunctions):
            return super().K(params, X, X2, **kwargs)
        _, log_levels, log_normalizer = self._log_weights(params)
        if (
            B.all(params["nu"] == np.inf)
            and self.num_levels == self.space.dimension + 1
        ):
            log_kernel = _log_hamming_graph_heat_kernel(
                params["lengthscale"], X, X2, q=getattr(self.space, "n_cat", 2)
            )
            return B.exp(log_kernel if self.normalize else log_kernel + log_normalizer)

        if self.normalize:
            log_levels = log_levels - B.max(log_levels)
            log_levels = log_levels - B.logsumexp(log_levels)
        weights = B.exp(log_levels)
        return self.eigenfunctions._weighted_outerproduct_from_level_weights(
            weights, X, X2
        )

    def K_diag(self, params: Dict[str, B.Numeric], X: B.Numeric, **kwargs) -> B.Numeric:
        if not isinstance(self.eigenfunctions, HammingGraphEigenfunctions):
            return super().K_diag(params, X, **kwargs)
        _, log_levels, log_normalizer = self._log_weights(params)
        diagonal = B.ones(B.dtype(log_levels), X.shape[0])
        # Retain a differentiation graph for the constant normalized diagonal.
        return (
            diagonal + 0.0 * log_normalizer
            if self.normalize
            else B.exp(log_normalizer) * diagonal
        )
