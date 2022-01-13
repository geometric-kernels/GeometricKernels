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

        params, state = self._kernel.init_params_and_state()

        self.lengthscale = gpflow.Parameter(params["lengthscale"], transform=positive())
        self.nu = gpflow.Parameter(params["nu"], transform=positive())
        self.state = state

    @property
    def space(self) -> Space:
        """Alias to kernel Space"""
        return self._kernel.space

    def K(self, X, X2=None):
        lengthscale = tf.convert_to_tensor(self.lengthscale)
        nu = tf.convert_to_tensor(self.nu)
        params = dict(lengthscale=lengthscale, nu=nu)
        return self._kernel.K(params, self.state, X, X2)

    def K_diag(self, X):
        lengthscale = tf.convert_to_tensor(self.lengthscale)
        nu = tf.convert_to_tensor(self.nu)
        params = dict(lengthscale=lengthscale, nu=nu)
        return self._kernel.K_diag(params, self.state, X)
