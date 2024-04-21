"""
(Approximate) finite-dimensional feature maps for geometric kernels.

A brief introduction into the theory can be found on :doc:`this page
</theory/feature_maps>`.
"""

# noqa: F401
from geometric_kernels.feature_maps.base import FeatureMap
from geometric_kernels.feature_maps.deterministic import DeterministicFeatureMapCompact
from geometric_kernels.feature_maps.random_phase import (
    RandomPhaseFeatureMapCompact,
    RandomPhaseFeatureMapNoncompact,
)
from geometric_kernels.feature_maps.rejection_sampling import (
    RejectionSamplingFeatureMapHyperbolic,
    RejectionSamplingFeatureMapSPD,
)
