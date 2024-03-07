"""
GPJax wrapper for `BaseGeometricKernel`
"""
import typing as tp
from dataclasses import dataclass

import gpjax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
from gpjax.base import param_field, static_field
from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.typing import Array, ScalarFloat
from jaxtyping import Float, Num

from ...kernels import BaseGeometricKernel

Kernel = tp.TypeVar("Kernel", bound="gpjax.kernels.base.AbstractKernel")  # noqa: F821


class GeometricKernelComputation(gpjax.kernels.computations.AbstractKernelComputation):
    """
    A class for computing the covariance matrix of a geometric kernel.
    """

    def cross_covariance(
        self,
        kernel: Kernel,
        x: Float[Array, "N #D1 D2"],  # noqa: F821
        y: Float[Array, "M #D1 D2"],  # noqa: F821
    ) -> Float[Array, "N M"]:
        """Compute the cross covariance matrix between two matrices of inputs.

        Args:
            x (Num[Array, "N #D1 D2"]): A batch of N inputs of, each of which
            is a matrix of size D1xD2, or a vector of size D2 if D1 is absent.
            y (Num[Array, "M #D1 D2"]): A batch of M inputs of, each of which
            is a matrix of size D1xD2, or a vector of size D2 if D1 is absent.

        Returns:
            Float[Array, "N M"]: The N x M covariance matrix.
        """
        return jnp.asarray(kernel(x, y))


@dataclass
class GPJaxGeometricKernel(gpjax.kernels.AbstractKernel):
    """
    GPJax wrapper for `BaseGeometricKernel`
    """

    nu: ScalarFloat = param_field(None, bijector=tfb.Softplus())
    lengthscale: tp.Union[ScalarFloat, Float[Array, " D"]] = param_field(
        None, bijector=tfb.Softplus()
    )
    variance: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    base_kernel: BaseGeometricKernel = static_field(None)
    compute_engine: AbstractKernelComputation = static_field(
        GeometricKernelComputation(), repr=False
    )
    name: str = "Geometric Kernel"

    def __post_init__(self):
        if self.base_kernel is None:
            raise ValueError("base_kernel must be specified")

        default_params = self.base_kernel.init_params()

        if self.nu is None:
            self.nu = jnp.array(default_params["nu"])

        if self.lengthscale is None:
            self.lengthscale = jnp.array(default_params["lengthscale"])

    def __call__(
        self, x: Num[Array, "N #D1 D2"], y: Num[Array, "M #D1 D2"]  # noqa: F821
    ) -> Float[Array, "N M"]:
        """Compute the cross covariance matrix between two matrices of inputs.

        Args:
            x (Num[Array, "N #D1 D2"]): A batch of N inputs of, each of which
            is a matrix of size D1xD2, or a vector of size D2 if D1 is absent.
            y (Num[Array, "M #D1 D2"]): A batch of M inputs of, each of which
            is a matrix of size D1xD2, or a vector of size D2 if D1 is absent.

        Returns:
            Float[Array, "N M"]: The N x M covariance matrix.
        """
        return self.variance * self.base_kernel.K(
            {"lengthscale": self.lengthscale, "nu": self.nu}, x, y
        )
