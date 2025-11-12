"""
GPJax kernel wrapper.

A tutorial on how to use this wrapper to run Gaussian process regression on
a geometric space is available in the
:doc:`frontends/GPJax.ipynb </examples/frontends/GPJax>` notebook.
"""

from dataclasses import dataclass

import gpjax
import jax.numpy as jnp
from beartype.typing import List, TypeVar, Union
from flax import nnx
from gpjax.kernels.computations.base import AbstractKernelComputation
from gpjax.linalg import Diagonal, psd
from gpjax.parameters import NonNegativeReal, PositiveReal
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
        x: Num[Array, "N #D1 D2"],  # noqa: F821
        y: Num[Array, "M #D1 D2"],  # noqa: F821
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
        nu_value = kernel.nu.value if kernel.trainable_nu else kernel.nu

        # Ensure inputs have `ndim` > 1. GPJax may squeeze shape `(1, 1)` into
        # `(1,)` which causes issues when passing to the base kernel.
        if x.ndim == 1:
            x = x[:, jnp.newaxis]
        if y.ndim == 1:
            y = y[:, jnp.newaxis]

        return kernel.variance.value * kernel.base_kernel.K(
            {"lengthscale": kernel.lengthscale.value, "nu": nu_value}, x, y
        )

    def diagonal(
        self, kernel: Kernel, x: Num[Array, "N #D1 D2"]  # noqa: F821
    ) -> Diagonal:
        """
        Compute the diagonal of the covariance matrix `K(x, x)` where `x` is a batch of
        vectors (or a batch of matrices) of inputs.

        Args:
            kernel:
                The kernel function.
            x:
                A batch of N inputs, each of which is a matrix of size D1xD2,
                or a vector of size D2 if D1 is absent.

        Returns:
            The computed diagonal variance as a `Diagonal` linear operator.
        """
        nu_value = kernel.nu.value if kernel.trainable_nu else kernel.nu

        # Ensure inputs have `ndim` > 1. GPJax may squeeze shape `(1, 1)` into
        # `(1,)` which causes issues when passing to the base kernel.
        if x.ndim == 1:
            x = x[:, jnp.newaxis]

        return psd(
            Diagonal(
                kernel.variance.value
                * kernel.base_kernel.K_diag(
                    {"lengthscale": kernel.lengthscale.value, "nu": nu_value}, x
                )
            )
        )


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

    :param base_kernel:
        The kernel to wrap.
    :param lengthscale:
        Initial value of the length scale.

        If not given or set to None, uses the default value of the
        `base_kernel`, as provided by its `init_params` method.
    :param nu:
        Initial value of the smoothness parameter nu.

        If not given or set to None, uses the default value of the
        `base_kernel`, as provided by its `init_params` method.
    :param variance:
        Initial value of the variance (outputscale) parameter.

        Defaults to 1.0.
    :param trainable_nu:
        Whether or not the parameter nu is to be optimized over.

        You must not change this parameter after constructing the object.
        Defaults to False.
    """

    nu: Union[ScalarFloat, nnx.Variable[ScalarFloat], None]
    lengthscale: nnx.Variable[Union[ScalarFloat, Float[Array, " D"]]]
    variance: nnx.Variable[ScalarFloat]

    base_kernel: BaseGeometricKernel
    compute_engine: AbstractKernelComputation = _GeometricKernelComputation()
    name: str = "Geometric Kernel"

    def __init__(
        self,
        base_kernel: BaseGeometricKernel,
        lengthscale: Union[
            Union[ScalarFloat, Float[Array, " D"]],
            nnx.Variable[Union[ScalarFloat, Float[Array, " D"]]],
            None,
        ] = None,
        nu: Union[ScalarFloat, nnx.Variable[ScalarFloat], None] = None,
        variance: Union[ScalarFloat, nnx.Variable[ScalarFloat]] = 1.0,
        trainable_nu: bool = False,
    ):
        active_dims = None
        n_dims = None
        super().__init__(active_dims, n_dims, self.compute_engine)

        self.base_kernel = base_kernel
        default_params = self.base_kernel.init_params()

        if lengthscale is None:
            lengthscale = jnp.array(default_params["lengthscale"])
        if nu is None:
            nu = jnp.array(default_params["nu"])

        if isinstance(lengthscale, nnx.Variable):
            self.lengthscale = lengthscale
        else:
            self.lengthscale = PositiveReal(lengthscale)

        self.trainable_nu = trainable_nu
        if not trainable_nu:
            self.nu = nu
        elif isinstance(nu, nnx.Variable):
            self.nu = nu
        else:
            self.nu = PositiveReal(nu)

        if isinstance(variance, nnx.Variable):
            self.variance = variance
        else:
            self.variance = NonNegativeReal(variance)

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
        return self.compute_engine.cross_covariance(self, x, y)
