"""
GPflow kernel wrapper
"""
from typing import Optional

import gpflow
import tensorflow as tf
from gpflow.base import TensorType
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
        base_kernel: BaseGeometricKernel,
        active_dims: Optional[ActiveDims] = None,
        name: Optional[str] = None,
        lengthscale: TensorType = None,
        nu: TensorType = None,
        variance: TensorType = 1.0,
        trainable_nu: bool = False,
    ):
        super().__init__(active_dims, name)
        self.base_kernel = base_kernel

        default_params = base_kernel.init_params()

        if nu is None:
            nu = default_params["nu"]

        if lengthscale is None:
            lengthscale = default_params["lengthscale"]

        self.lengthscale = gpflow.Parameter(lengthscale, transform=positive())
        self.variance = gpflow.Parameter(variance, transform=positive())

        self.trainable_nu = trainable_nu
        if self.trainable_nu and tf.math.is_inf(nu):
            raise ValueError("Cannot have trainable `nu` parameter with infinite value")

        if self.trainable_nu:
            self.nu = gpflow.Parameter(nu, transform=positive())
        else:
            self.nu = nu

    @property
    def space(self) -> Space:
        """Alias to kernel Space"""
        return self.base_kernel.space

    def K(self, X, X2=None):
        lengthscale = tf.convert_to_tensor(self.lengthscale)
        nu = tf.cast(tf.convert_to_tensor(self.nu), lengthscale.dtype)
        variance = tf.convert_to_tensor(self.variance)
        params = dict(lengthscale=lengthscale, nu=nu)
        return variance*self.base_kernel.K(params, X, X2)

    def K_diag(self, X):
        lengthscale = tf.convert_to_tensor(self.lengthscale)
        nu = tf.cast(tf.convert_to_tensor(self.nu), lengthscale.dtype)
        variance = tf.convert_to_tensor(self.variance)
        params = dict(lengthscale=lengthscale, nu=nu)
        return variance*self.base_kernel.K_diag(params, X)


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
