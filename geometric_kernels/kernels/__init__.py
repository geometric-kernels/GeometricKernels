"""
This module provides the abstract base class for geometric kernels and
specialized classes for various types of spaces.

Unless you know exactly what you are doing, always use the
:class:`MaternGeometricKernel` that "just works".
"""

# noqa: F401
from geometric_kernels.kernels.base import BaseGeometricKernel
from geometric_kernels.kernels.feature_map import MaternFeatureMapKernel
from geometric_kernels.kernels.karhunen_loeve import MaternKarhunenLoeveKernel
from geometric_kernels.kernels.matern_kernel import (
    MaternGeometricKernel,
    default_feature_map,
)
from geometric_kernels.kernels.product import ProductGeometricKernel
