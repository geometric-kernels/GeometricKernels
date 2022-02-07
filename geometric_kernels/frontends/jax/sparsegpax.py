"""
SparseGPax kernel wrapper
"""
from typing import NamedTuple

import jax.numpy as jnp
import sparsegpax.kernel

from geometric_kernels.kernels import BaseGeometricKernel


class GeometricKernelParameters(NamedTuple):
    log_lengthscale: jnp.ndarray
    log_nu: jnp.ndarray


class SparseGPaxGeometricKernel(sparsegpax.kernel.AbstractKernel):
    """
    SparseGPax wrapper for `BaseGeometricKernel`
    """

    def __init__(self, kernel: BaseGeometricKernel):
        self._kernel = kernel
        self._init_params, self._state = kernel.init_params_and_state()

    def init_params(self, key) -> GeometricKernelParameters:
        params = self._init_params

        return GeometricKernelParameters(
            log_lengthscale=jnp.log(params["lengthscale"]), log_nu=jnp.log(params["nu"])
        )

    def matrix(self, params: GeometricKernelParameters, x1, x2):
        kernel_params = {
            "lengthscale": jnp.exp(params.log_lengthscale),
            "nu": jnp.exp(params.log_nu),
        }

        return self._kernel.K(kernel_params, self._state, x1, x2)

    def kernel(self, params: GeometricKernelParameters):
        return self._kernel

    def standard_spectral_measure(self, key, num_samples):
        raise NotImplementedError("Spectral measure not implemented")

    def spectral_weights(self, params, frequency):
        raise NotImplementedError("Spectral weigths not implemented")
