"""
Samplers.
"""
from __future__ import annotations  # By https://stackoverflow.com/a/62136491

from functools import partial

import lab as B
from beartype.typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

# By https://stackoverflow.com/a/62136491
if TYPE_CHECKING:
    from geometric_kernels.feature_maps import FeatureMap


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

    key, random_weights = B.randn(
        key, B.dtype(params["lengthscale"]), num_features, s
    )  # [M, S]

    random_sample = B.matmul(features, random_weights)  # [N, S]

    return key, random_sample


def sampler(
    feature_map: FeatureMap, s: Optional[int] = 1, **kwargs
) -> Callable[[Any], Any]:
    """
    A helper wrapper around `sample_at`.

    Given a `feature_map`, return a function that computes `s` samples.
    """

    sample_f = partial(sample_at, feature_map, s, **kwargs)
    sample_f.__doc__ == sample_at.__doc__

    return sample_f
