"""Log-domain random-phase features for hypercube and Hamming graphs."""

import lab as B
import numpy as np
from beartype.typing import Dict, Tuple, Union

from geometric_kernels.feature_maps.random_phase import RandomPhaseFeatureMapCompact
from geometric_kernels.lab_extras import from_numpy, is_complex
from geometric_kernels.spaces import HammingGraph, HypercubeGraph


class RandomPhaseFeatureMapHammingGraph(RandomPhaseFeatureMapCompact):
    """Random-phase features with log-domain weighting for Hamming graphs.

    Supports both :class:`~.spaces.HammingGraph` and
    :class:`~.spaces.HypercubeGraph`. Combines log spectral weights, log
    multiplicities, and normalized Kravchuk magnitudes before exponentiation.
    Signs and zero polynomial values are preserved.

    Sampling, feature ordering, and row normalization follow
    :class:`RandomPhaseFeatureMapCompact`. The log-domain computation prevents
    premature coefficient underflow and multiplicity overflow, but does not
    eliminate Kravchuk recurrence error or cancellation.

    :param space:
        A Hamming graph or binary hypercube graph.
    :param num_levels:
        Number of spectral levels to include.
    :param num_random_phases:
        Number of sampled phases. The map returns this many features per level.
    """

    def __init__(
        self,
        space: Union[HammingGraph, HypercubeGraph],
        num_levels: int,
        num_random_phases: int = 3000,
    ):
        if not isinstance(space, (HammingGraph, HypercubeGraph)):
            raise ValueError("space must be a HammingGraph or HypercubeGraph")
        super().__init__(space, num_levels, num_random_phases)

    def __call__(
        self,
        X: B.Numeric,
        params: Dict[str, B.Numeric],
        *,
        key: B.RandomState,
        normalize: bool = True,
        **kwargs,
    ) -> Tuple[B.RandomState, B.Numeric]:
        """Return the updated random key and log-domain random-phase features.

        Arguments and return shapes follow ``RandomPhaseFeatureMapCompact``.
        Normalization produces unit-norm feature rows; unnormalized features
        can still exceed the floating-point range.
        """
        from geometric_kernels.kernels.matern_kernel_hamming_graph import (
            MaternKernelHammingGraph,
        )

        key, phases = self.space.random(key, self.num_random_phases)
        log_spectrum = MaternKernelHammingGraph.log_spectrum(
            self.space.get_eigenvalues(self.num_levels),
            params["nu"],
            params["lengthscale"],
            self.space.dimension,
        )
        phases = B.cast(B.dtype(X), from_numpy(X, phases))
        return key, self._features_from_log_spectrum(log_spectrum, X, phases, normalize)

    def _features_from_log_spectrum(self, log_spectrum, X, phases, normalize):
        polynomials = self.eigenfunctions.phi_product_normalized(
            X, phases, dtype=B.dtype(log_spectrum)
        )
        log_multiplicities = B.cast(
            B.dtype(log_spectrum),
            from_numpy(
                log_spectrum, self.eigenfunctions.log_num_eigenfunctions_per_level
            ),
        )[:, None]
        nonzero = polynomials != 0
        # Avoid log(0) in the differentiation graph, including masked branches.
        safe_abs = B.where(nonzero, B.abs(polynomials), B.ones(polynomials))
        log_magnitude = B.log(safe_abs) + B.transpose(
            0.5 * log_spectrum + log_multiplicities
        )
        log_magnitude = B.where(nonzero, log_magnitude, -np.inf)
        log_magnitude = B.reshape(log_magnitude, X.shape[0], -1)
        signs = B.reshape(B.sign(polynomials), X.shape[0], -1)
        if normalize:
            log_magnitude = log_magnitude - B.max(log_magnitude, axis=1, squeeze=False)
            log_magnitude = log_magnitude - 0.5 * B.logsumexp(
                2 * log_magnitude, axis=1, squeeze=False
            )
        features = signs * B.exp(log_magnitude)
        if is_complex(X):
            features = B.concat(features, B.zeros(features), axis=1)
        return features
