"""
Deterministic Fourier features sampler.
"""

import lab as B

from geometric_kernels.sampling.base import BaseSampler


class FourierSampler(BaseSampler):
    """
    A sampler that uses Fourier (...) features to sample from a GP.
    """

    def sample(self, feature_map, X, s=1, key=None, **kwargs):

        features = feature_map(X)  # [N, M]

        key = key or B.global_random_state(B.dtype(X))

        num_features = B.shape(features)[-1]

        key, random_weights = B.randn(key, B.dtype(X), num_features, s)  # [M, S]

        random_sample = B.matmul(features, random_weights)  # [N, S]

        return key, random_sample
