"""
This module provides the :class:`RejectionSamplingFeatureMapHyperbolic` and the
:class:`RejectionSamplingFeatureMapSPD`, rejection sampling-based feature maps
for :class:`Hyperbolic` and :class:`SymmetricPositiveDefiniteMatrices`, resp.
"""

import lab as B

from geometric_kernels.feature_maps.base import FeatureMap
from geometric_kernels.feature_maps.probability_densities import (
    hyperbolic_density_sample,
    spd_density_sample,
)
from geometric_kernels.lab_extras import from_numpy
from geometric_kernels.spaces import Hyperbolic, SymmetricPositiveDefiniteMatrices


class RejectionSamplingFeatureMapHyperbolic(FeatureMap):
    def __init__(self, space: Hyperbolic, num_random_phases: int = 3000):
        """
        Random phase feature map for the Hyperbolic space based on the
        rejection sampling algorithm.

        :param space: Hyperbolic space.
        :param num_random_phases: number of random phases to use.
        """
        self.space = space
        self.num_random_phases = num_random_phases

    def __call__(
        self, X: B.Numeric, params, *, key, normalize=True, **kwargs
    ) -> B.Numeric:
        """
        :param X: [N, D] points in the space to evaluate the map on.
        :param params: parameters of the feature map (lengthscale and smoothness).
        :param key: random state, either `np.random.RandomState`, `tf.random.Generator`,
                    `torch.Generator` or `jax.tensor` (representing random state).

                     Note that for any backend other than `jax`, passing the same `key`
                     twice does not guarantee that the feature map will be the same each time.
                     This is because these backends' random state has... a state.
                     One either has to recreate/restore the state each time or
                     make use of `geometric_kernels.utils.make_deterministic`.
        :param normalize: normalize to have unit average variance (`True` by default).
        :param ``**kwargs``: unused.

        :return: `Tuple(key, features)` where `features` is [N, O] features,
                 and `key` is the new key for `jax`, and the same random
                 state (generator) for all other backends.
        """
        # Default behavior
        if normalize is None:
            normalize = True

        key, random_phases = self.space.random_phases(
            key, self.num_random_phases
        )  # [O, D]

        key, random_lambda = hyperbolic_density_sample(
            key,
            (self.num_random_phases, B.rank(self.space.rho)),
            params,
            self.space.dimension,
        )  # [O, 1]

        random_phases_b = B.expand_dims(
            B.cast(B.dtype(params["lengthscale"]), from_numpy(X, random_phases))
        )  # [1, O, D]
        random_lambda_b = B.expand_dims(
            B.cast(B.dtype(params["lengthscale"]), from_numpy(X, random_lambda))
        )  # [1, O, 1]
        X_b = B.expand_dims(X, axis=-2)  # [N, 1, D]

        p = self.space.power_function(random_lambda_b, X_b, random_phases_b)  # [N, O]

        features = B.concat(B.real(p), B.imag(p), axis=-1)  # [N, 2*O]
        if normalize:
            normalizer = B.sqrt(B.sum(features**2, axis=-1, squeeze=False))
            features = features / normalizer

        return key, features


class RejectionSamplingFeatureMapSPD(FeatureMap):
    def __init__(
        self,
        space: SymmetricPositiveDefiniteMatrices,
        num_random_phases: int = 3000,
    ):
        """
        Random phase feature map for the SPD (Symmetric Positive Definite) space based on the
        rejection sampling algorithm.

        :param space: SymmetricPositiveDefiniteMatrices space.
        :param num_random_phases: number of random phases to use.
        """
        self.space = space
        self.num_random_phases = num_random_phases

    def __call__(
        self, X: B.Numeric, params, *, key, normalize=True, **kwargs
    ) -> B.Numeric:
        """
        :param X: [N, D, D] points in the space to evaluate the map on.
        :param params: parameters of the feature map (lengthscale and smoothness).
        :param key: random state, either `np.random.RandomState`, `tf.random.Generator`,
                    `torch.Generator` or `jax.tensor` (representing random state).

                     Note that for any backend other than `jax`, passing the same `key`
                     twice does not guarantee that the feature map will be the same each time.
                     This is because these backends' random state has... a state.
                     One either has to recreate/restore the state each time or
                     make use of `geometric_kernels.utils.make_deterministic`.
        :param normalize: normalize to have unit average variance (`True` by default).
        :param ``**kwargs``: unused.

        :return: `Tuple(key, features)` where `features` is [N, O] features,
                 and `key` is the new key for `jax`, and the same random
                 state (generator) for all other backends.
        """
        # Default behavior for normalization
        if normalize is None:
            normalize = True

        key, random_phases = self.space.random_phases(
            key, self.num_random_phases
        )  # [O, D, D]

        key, random_lambda = spd_density_sample(
            key, (self.num_random_phases,), params, self.space.degree, self.space.rho
        )  # [O, D]

        random_phases_b = B.expand_dims(
            B.cast(B.dtype(params["lengthscale"]), from_numpy(X, random_phases))
        )  # [1, O, D, D]
        random_lambda_b = B.expand_dims(
            B.cast(B.dtype(params["lengthscale"]), from_numpy(X, random_lambda))
        )  # [1, O, D]
        X_b = B.expand_dims(X, axis=-3)  # [N, 1, D, D]

        p = self.space.power_function(random_lambda_b, X_b, random_phases_b)  # [N, O]

        features = B.concat(B.real(p), B.imag(p), axis=-1)  # [N, 2*O]
        if normalize:
            normalizer = B.sqrt(B.sum(features**2, axis=-1, squeeze=False))
            features = features / normalizer

        return key, features
