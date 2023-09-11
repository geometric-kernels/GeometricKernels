import typing as tp

import gpjax
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

from ...kernels import BaseGeometricKernel


class _GeometricComputation(gpjax.kernels.AbstractKernelComputation):
    """
    A class for computing the covariance matrix of a geometric kernel.
    """

    def __init__(
        self,
        kernel_fn: tp.Callable[
            [tp.Dict, Float[Array, "N D"], Float[Array, "M D"]], Array
        ] = None,
    ) -> None:
        """Initialise the computation class.

        Args:
            kernel_fn (tp.Callable, optional): A kernel function that accepts a pair of matrics of lengths N and M, and returns the NxM covariance matrix. Defaults to None.
        """
        super().__init__(kernel_fn)

    def cross_covariance(
        self, params: tp.Dict, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        """Compute the cross covariance matrix between two matrices of inputs.

        Args:
            params (tp.Dict): The dictionary of parameters used to compute the covariance matrix.
            x (Float[Array, "N D"]): An N x D matrix of inputs.
            y (Float[Array, "M D"]): An M x D matrix of inputs.

        Returns:
            Float[Array, "N M"]: The N x M covariance matrix.
        """
        matrix = jnp.asarray(self.kernel_fn(params, x, y))
        return matrix


class GPJaxGeometricKernel(gpjax.kernels.AbstractKernel):
    """A class for wrapping a geometric kernel in a GPJax-compatible format."""

    def __init__(
        self,
        base_kernel: BaseGeometricKernel,
        compute_engine=_GeometricComputation,
        active_dims: tp.Optional[tp.List[int]] = None,
        name: tp.Optional[str] = "Geometric Kernel",
    ) -> None:
        """Initialise the kernel.

        Args:
            base_kernel (BaseGeometricKernel): The geometric kernel to wrap.
            compute_engine (_GeometricComputation, optional): The compute engine that assigns the logic used to compute covariance matrices. Defaults to _GeometricComputation.
            active_dims (tp.Optional[tp.List[int]], optional): The indices of the inputs data to use. Defaults to None.
            name (tp.Optional[str], optional): Kernel name. Defaults to "Geometric Kernel".
        """
        super().__init__(compute_engine, active_dims, True, False, name)
        self.base_kernel = base_kernel

    def __call__(
        self, params: tp.Dict, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N D"]:
        """Compute the cross covariance matrix between two matrices of inputs.

        Args:
            params (tp.Dict): The dictionary of parameters used to compute the covariance matrix.
            x (Float[Array, "N D"]): An N x D matrix of inputs.
            y (Float[Array, "M D"]): An M x D matrix of inputs.

        Returns:
            Float[Array, "N M"]: The N x M covariance matrix.
        """
        return self.base_kernel.K(params, self.state, x, y)

    def init_params(self, key: jr.KeyArray = None) -> tp.Dict:
        """Initialise the parameters of the kernel.

        Args:
            key (jr.KeyArray, optional): PRNGKey that is passed around during initialisation to initialise any stochastic parameters. Defaults to None.

        Returns:
            tp.Dict: A dictionary of parameters
        """
        params = self.base_kernel.init_params()
        # Convert each value to Jax arrays
        return jax.tree_util.tree_map(lambda x: jnp.atleast_1d(x), params)
