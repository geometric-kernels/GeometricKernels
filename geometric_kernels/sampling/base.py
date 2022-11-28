"""
Base class for samplers
"""
import abc
from typing import Generic, TypeVar

import lab as B

from geometric_kernels.kernels import BaseGeometricKernel
from geometric_kernels.types import FeatureMap

T = TypeVar("T", bound=BaseGeometricKernel)


class BaseSampler(abc.ABC, Generic[T]):
    """
    Abstract base class for samplers.
    """

    def __init__(self, kernel: T):
        self._kernel = kernel

    @property
    def kernel(self) -> T:
        return self._kernel

    @abc.abstractmethod
    def sample(
        self, feature_map: FeatureMap, X: B.Numeric, s=1, key=None, **kwargs,
    ):
        """
        Draw `s` random samples determined by `key` from a GP determined by kernel's `feature_map` in points `X` with additional `**kwargs` attributes.
        """
        raise NotImplementedError
