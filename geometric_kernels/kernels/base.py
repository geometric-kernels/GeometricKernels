"""
Base class for geometric kernels
"""

import abc

import lab as B
from beartype.typing import Generic, TypeVar

from geometric_kernels.spaces import Space

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
    def init_params(self):
        """
        Initializes the dict of the trainable parameters of the kernel. In
        (almost) all cases, it contains two keys: `"nu"` and `"lengthscale"`.
        This dict can be modified and is passed around into such methods as `K`
        or `K_diag`, as the `params` argument.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def K(self, params, X, X2=None, **kwargs) -> B.Numeric:
        """
        Returns pairwise covariance between `X` and `X2`.

        .. note::
           The types of values in the `params` dict determine the backend
           used for internal computations and the output type.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def K_diag(self, params, X, **kwargs) -> B.Numeric:
        """
        Returns the diagonal of `self.K(params, X, X)`.

        .. note::
           The types of values in the `params` dict determine the backend
           used for internal computations and the output type.
        """
        raise NotImplementedError
