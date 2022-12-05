"""
Samplers.
"""
from functools import partial
from typing import Any, Callable, Optional, Tuple

import lab as B

from geometric_kernels.types import FeatureMap


def sampler(
    feature_map: FeatureMap, s: Optional[int] = 1, key: Optional[Any] = None, **kwargs
) -> Callable[[Any], Any]:
    """
    Given a `feature_map`, return a function that computes `s` samples with `key` random state at given points in space.
    """

    def _sample(feature_map, s, key, X: B.Numeric) -> Tuple[Any, Any]:
        features = feature_map(X)  # [N, M]

        key = key or B.global_random_state(B.dtype(X))

        num_features = B.shape(features)[-1]

        key, random_weights = B.randn(key, B.dtype(X), num_features, s)  # [M, S]

        random_sample = B.matmul(features, random_weights)  # [N, S]

        return key, random_sample

    sample_f = partial(_sample, feature_map, s, key)
    sample_f.__doc__ == "Compute random samples at `X`"

    return sample_f
