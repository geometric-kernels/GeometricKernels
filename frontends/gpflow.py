from kernels.geometric_kernel import BaseGeometricKernel

import gpflow
import tensorflow as tf
import tensorlfow_probability as tfp


class GeometricKernel(gpflow.kernels.Kernel):
    def __init__(self, space, nu, *args, **kwargs):
        self._underlying_kernel = BaseGeometricKernel(space, nu,
                                                      *args, **kwargs)
        self.lengthscale = gpflow.Parameter(tf.convert_to_tensor([1.0]),
                                            trainable=True,
                                            transform=tfp.bijectors.Exp())
        super().__init__()

    @property
    def nu(self):
        return self._underlying_kernel.nu

    @property
    def space(self):
        return self._underlying_kernel.space

    def K(self, X, X2=None):
        return self._underlying_kernel(self.lengthscale.read_value(),
                                       X, X2)

    def K_diag(self, X):
        return self._underlying_kernel(self.lengthscale.read_value(),
                                       X)
