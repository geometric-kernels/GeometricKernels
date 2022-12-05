"""
Base class for geometric kernels
"""
import abc
from typing import Any, Dict, Generic, Tuple, TypeVar

import lab as B

from geometric_kernels.spaces import Space
from geometric_kernels.types import FeatureMap

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
    def init_params_and_state(self):
        """
        Returns initial parameters and state of the kernels.
        params is a dict of trainable parameters of the kernel, such as lengthscale.
        state is a dict non-trainable parameters of the kernel.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def K(self, params, state, X, X2=None, **kwargs) -> B.Numeric:
        """
        Returns pairwise covariance between `X` and `X2`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def K_diag(self, params, state, X, **kwargs) -> B.Numeric:
        """
        Returns covariance between elements in `X`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def feature_map(self, params, state, **kwargs) -> Tuple[FeatureMap, Dict[str, Any]]:
        """
        Returns an (approximate) feature map and some context (e.g. random state used to generate random features).
        """
