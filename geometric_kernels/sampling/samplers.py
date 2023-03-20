"""
Samplers.
"""
from functools import partial
from typing import Any, Callable, Optional, Tuple

import lab as B

from geometric_kernels.types import FeatureMap


def sample_at(feature_map, s, X: B.Numeric, params, state, key=None) -> Tuple[Any, Any]:
    """
    Given a `feature_map`, compute `s` samples at `X` defined by random state `key`.
    """

    key = key or B.global_random_state(B.dtype(X))

    features, _context = feature_map(X, params, state, key=key)  # [N, M]

    if "key" in _context:
        key = _context["key"]

    num_features = B.shape(features)[-1]

    key, random_weights = B.randn(key, B.dtype(X), num_features, s)  # [M, S]

    random_sample = B.matmul(features, random_weights)  # [N, S]

    return key, random_sample


def sampler(
    feature_map: FeatureMap, s: Optional[int] = 1, **kwargs
) -> Callable[[Any], Any]:
    """
    A helper wrapper around `sample_at`.

    Given a `feature_map`, return a function that computes `s` samples with `key` random state at given points in space.
    """

    sample_f = partial(sample_at, feature_map, s)
    sample_f.__doc__ == sample_at.__doc__

    return sample_f
