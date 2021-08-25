"""
Base class for geometric kernels
"""
import abc
from typing import Callable, Generic, List, Optional, TypeVar

import numpy as np
import tensorflow as tf

from geometric_kernels.spaces import Mesh, Space
from geometric_kernels.types import Parameter, TensorLike

T = TypeVar("T", bound=Space)
"""
Type bounding `Space`, used as Generic type in `BaseGeometricKernel`.
"""


class BaseGeometricKernel(abc.ABC, Generic[T]):
    """
    Abstract base class for Geometric kernels.
    """

    def __init__(self, space: T):
        self._space = space

    @property
    def space(self) -> T:
        """
        Returns space on which the kernel is defined.
        """
        return self._space

    @abc.abstractmethod
    def K(self, X, X2=None, **kwargs) -> TensorLike:
        """
        Returns pairwise covariance between `X` and `X2`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def K_diag(self, X, **kwargs) -> TensorLike:
        """
        Returns covariance between elements in `X`.
        """
        raise NotImplementedError
