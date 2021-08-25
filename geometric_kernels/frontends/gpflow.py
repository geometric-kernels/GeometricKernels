"""
GPflow kernel wrapper
"""
from typing import Optional

import gpflow
import tensorflow as tf
from gpflow.kernels.base import ActiveDims
from gpflow.utilities import positive

from geometric_kernels.kernels import BaseGeometricKernel
from geometric_kernels.spaces.base import Space


class GPflowGeometricKernel(gpflow.kernels.Kernel):
    """
    GPflow wrapper for `BaseGeometricKernel`.
    """

    def __init__(
        self,
        kernel: BaseGeometricKernel,
        active_dims: Optional[ActiveDims] = None,
        name: Optional[str] = None,
    ):
        super().__init__(active_dims, name)
        self._kernel = kernel
        self.lengthscale = gpflow.Parameter(1.0, transform=positive())

    @property
    def space(self) -> Space:
        """Alias to kernel Space"""
        return self._kernel.space

    def K(self, X, X2=None):
        return self._kernel.K(X, X2, lengthscale=self.lengthscale)

    def K_diag(self, X):
        return self._kernel.K_diag(X, lengthscale=self.lengthscale)
