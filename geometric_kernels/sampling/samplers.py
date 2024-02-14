"""
Samplers.
"""
from functools import partial
from typing import Any, Callable, Optional, Tuple

import lab as B

from geometric_kernels.types import FeatureMap


def sample_at(
    feature_map, s, X: B.Numeric, params, key=None, normalize=None
) -> Tuple[Any, Any]:
    """
    Given a `feature_map`, compute `s` samples at `X` defined by random state `key`.
    """

    if key is None:
        key = B.global_random_state(B.dtype(X))

    _context, features = feature_map(X, params, key=key, normalize=normalize)  # [N, M]

    if _context is not None:
        key = _context

    num_features = B.shape(features)[-1]

    key, random_weights = B.randn(key, B.dtype(params["lengthscale"]), num_features, s)  # [M, S]

    random_sample = B.matmul(features, random_weights)  # [N, S]

    return key, random_sample


def sampler(
    feature_map: FeatureMap, s: Optional[int] = 1, key=None, **kwargs
) -> Callable[[Any], Any]:
    """
    A helper wrapper around `sample_at`.

    Given a `feature_map`, return a function that computes `s` samples with `key` random state at given points in space.
    """

    sample_f = partial(sample_at, feature_map, s, key=key)
    sample_f.__doc__ == sample_at.__doc__

    return sample_f
