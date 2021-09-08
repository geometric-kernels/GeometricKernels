"""
GPflow kernel wrapper
"""
from typing import Optional

import tensorflow as tf

import gpflow
from geometric_kernels.kernels import BaseGeometricKernel
from geometric_kernels.spaces.base import Space
from gpflow.kernels.base import ActiveDims
from gpflow.utilities import positive


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
        lengthscale = tf.convert_to_tensor(self.lengthscale)
        return self._kernel.K(X, X2, lengthscale=lengthscale)

    def K_diag(self, X):
        lengthscale = tf.convert_to_tensor(self.lengthscale)
        return self._kernel.K_diag(X, lengthscale=lengthscale)
