"""
GPJax kernel wrapper.

A tutorial on how to use this wrapper to run Gaussian process regression on
a geometric space is available in the
:doc:`frontends/GPJax.ipynb </examples/frontends/GPJax>` notebook.
"""

from dataclasses import dataclass

import gpjax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
from beartype.typing import List, TypeVar, Union
from gpjax.base import param_field, static_field
from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.typing import Array, ScalarFloat
from jaxtyping import Float, Num

from geometric_kernels.kernels import BaseGeometricKernel
from geometric_kernels.spaces import Space

Kernel = TypeVar("Kernel", bound="gpjax.kernels.base.AbstractKernel")  # noqa: F821


class _GeometricKernelComputation(gpjax.kernels.computations.AbstractKernelComputation):
    """
    A class for computing the covariance matrix of a geometric kernel.
    """

    def cross_covariance(
        self,
        kernel: Kernel,
        x: Float[Array, "N #D1 D2"],  # noqa: F821
        y: Float[Array, "M #D1 D2"],  # noqa: F821
    ) -> Float[Array, "N M"]:
        """
        Compute the cross covariance matrix between two batches of vectors (or
        batches of matrices) of inputs.

        :param x:
            A batch of N inputs, each of which is a matrix of size D1xD2,
              or a vector of size D2 if D1 is absent.
        :param y:
            A batch of M inputs, each of which is a matrix of size D1xD2,
              or a vector of size D2 if D1 is absent.

        :return:
            The N x M covariance matrix.
        """
        return jnp.asarray(kernel(x, y))


@dataclass
class GPJaxGeometricKernel(gpjax.kernels.AbstractKernel):
    r"""
    GPJax wrapper for :class:`~.kernels.BaseGeometricKernel`.

    A tutorial on how to use this wrapper to run Gaussian process regression on
    a geometric space is available in the
    :doc:`frontends/GPJax.ipynb </examples/frontends/GPJax>` notebook.

    .. note::
        Remember that the `base_kernel` itself does not store any of its
        hyperparameters (like `lengthscale` and `nu`). If you do not set them
        manually—when initializing the object or after, by setting the
        properties—this wrapper will use the values provided by
        `base_kernel.init_params`.

    .. note::
        Unlike the frontends for GPflow and GPyTorch, GPJaxGeometricKernel
        does not have the `trainable_nu` parameter which determines whether or
        not the smoothness parameter nu is to be optimized over. By default, it
        is not trainable. If you want to make it trainable, do
        :code:`kernel = kernel.replace_trainable(nu=False)` on an instance of
        the `GPJaxGeometricKernel`.

    :param base_kernel:
        The kernel to wrap.
    :type base_kernel: geometric_kernels.kernels.BaseGeometricKernel
    :param name:
        Optional kernel name (inherited from `gpjax.kernels.AbstractKernel`).

        Defaults to "Geometric Kernel".
    :type name: str
    :param lengthscale:
        Initial value of the length scale.

        If not given or set to None, uses the default value of the
        `base_kernel`, as provided by its `init_params` method.
    :type lengthscale: Union[ScalarFloat, Float[Array, " D"]]
    :param nu:
        Initial value of the smoothness parameter nu.

        If not given or set to None, uses the default value of the
        `base_kernel`, as provided by its `init_params` method.
    :type nu: ScalarFloat
    :param variance:
        Initial value of the variance (outputscale) parameter.

        Defaults to 1.0.
    :type variance: ScalarFloat
    """

    nu: ScalarFloat = param_field(None, bijector=tfb.Softplus(), trainable=False)
    lengthscale: Union[ScalarFloat, Float[Array, " D"]] = param_field(
        None, bijector=tfb.Softplus()
    )
    variance: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    base_kernel: BaseGeometricKernel = static_field(None)
    compute_engine: AbstractKernelComputation = static_field(
        _GeometricKernelComputation(), repr=False
    )
    name: str = "Geometric Kernel"

    def __post_init__(self):
        if self.base_kernel is None:
            raise ValueError("base_kernel must be specified")

        default_params = self.base_kernel.init_params()

        if self.nu is None:
            self.nu = jnp.array(default_params["nu"])
        if isinstance(self.nu, ScalarFloat):
            self.nu = jnp.array([self.nu])

        if self.lengthscale is None:
            self.lengthscale = jnp.array(default_params["lengthscale"])
        if isinstance(self.lengthscale, ScalarFloat):
            self.lengthscale = jnp.array([self.lengthscale])

    @property
    def space(self) -> Union[Space, List[Space]]:
        r"""Alias to the `base_kernel`\ s space property."""
        return self.base_kernel.space

    def __call__(
        self, x: Num[Array, "N #D1 D2"], y: Num[Array, "M #D1 D2"]  # noqa: F821
    ) -> Float[Array, "N M"]:
        """
        Compute the cross-covariance matrix between two batches of vectors (or
        batches of matrices) of inputs.

        :param x:
            A batch of N inputs, each of which is a matrix of size D1xD2,
            or a vector of size D2 if D1 is absent.
        :param y:
            A batch of M inputs, each of which is a matrix of size D1xD2,
            or a vector of size D2 if D1 is absent.

        :return:
            The N x M cross-covariance matrix.
        """
        return self.variance * self.base_kernel.K(
            {"lengthscale": self.lengthscale, "nu": self.nu}, x, y
        )
