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

        params = self._kernel.init_params()

        self.lengthscale = gpflow.Parameter(params["lengthscale"], transform=positive())
        self.nu = gpflow.Parameter(params["nu"], transform=positive())

    @property
    def space(self) -> Space:
        """Alias to kernel Space"""
        return self._kernel.space

    def K(self, X, X2=None):
        lengthscale = tf.convert_to_tensor(self.lengthscale)
        nu = tf.convert_to_tensor(self.nu)
        params = dict(lengthscale=lengthscale, nu=nu)
        return self._kernel.K(params, X, X2)

    def K_diag(self, X):
        lengthscale = tf.convert_to_tensor(self.lengthscale)
        nu = tf.convert_to_tensor(self.nu)
        params = dict(lengthscale=lengthscale, nu=nu)
        return self._kernel.K_diag(params, X)


class DefaultFloatZeroMeanFunction(gpflow.mean_functions.Constant):
    """
    Zero mean function. The default GPflow `ZeroMeanFunction`
    uses the input's dtype as output type, this minor adaptation
    uses GPflow's `default_float` instead.
    """

    def __init__(self, output_dim=1):
        super().__init__()
        self.output_dim = output_dim
        del self.c

    def __call__(self, inputs):
        output_shape = tf.concat([tf.shape(inputs)[:-1], [self.output_dim]], axis=0)
        return tf.zeros(output_shape, dtype=gpflow.default_float())
