"""
GPflow kernel wrapper.

A tutorial on how to use this wrapper to run Gaussian process regression on
a geometric space is available in the
:doc:`frontends/GPflow.ipynb </examples/frontends/GPflow>` notebook.
"""

import gpflow
import numpy as np
import tensorflow as tf
from beartype.typing import List, Optional, Union
from gpflow.base import TensorType
from gpflow.kernels.base import ActiveDims
from gpflow.utilities import positive

from geometric_kernels.kernels import BaseGeometricKernel
from geometric_kernels.spaces import Space


class GPflowGeometricKernel(gpflow.kernels.Kernel):
    r"""
    GPflow wrapper for :class:`~.kernels.BaseGeometricKernel`.

    A tutorial on how to use this wrapper to run Gaussian process regression on
    a geometric space is available in the
    :doc:`frontends/GPflow.ipynb </examples/frontends/GPflow>` notebook.

    .. note::
        Remember that the `base_kernel` itself does not store any of its
        hyperparameters (like `lengthscale` and `nu`). If you do not set them
        manually—when initializing the object or after, by setting the
        properties—this wrapper will use the values provided by
        `base_kernel.init_params`.

    .. note::
        As customary in GPflow, this wrapper calls the length scale
        parameter `lengthscales` (plural), as opposed to the convention used by
        GeometricKernels, where we call it `lengthscale` (singular).

    :param base_kernel:
        The kernel to wrap.
    :param active_dims:
        Active dimensions, either a slice or list of indices into the
        columns of X (inherited from `gpflow.kernels.base.Kernel`).
    :param name:
        Optional kernel name (inherited from `gpflow.kernels.base.Kernel`).
    :param lengthscales:
        Initial value of the length scale. Note **s** in lengthscale\ **s**\ .

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

        Cannot be True if nu is equal to infinity. You cannot change
        this parameter after constructing the object. Defaults to False.

    :raises ValueError:
        If trying to set nu = infinity together with trainable_nu = True.
    """

    def __init__(
        self,
        base_kernel: BaseGeometricKernel,
        active_dims: Optional[ActiveDims] = None,
        name: Optional[str] = None,
        lengthscales: Union[float, TensorType, np.ndarray] = None,
        nu: Union[float, TensorType, np.ndarray] = None,
        variance: Union[float, TensorType, np.ndarray] = 1.0,
        trainable_nu: bool = False,
    ):
        super().__init__(active_dims, name)
        self.base_kernel = base_kernel

        default_params = base_kernel.init_params()

        if nu is None:
            nu = default_params["nu"]
        if type(nu) is float:
            nu = np.array([nu])

        if lengthscales is None:
            lengthscales = default_params["lengthscale"]
        if type(lengthscales) is float:
            lengthscales = np.array([lengthscales])

        self.lengthscales = gpflow.Parameter(lengthscales, transform=positive())
        self.variance = gpflow.Parameter(variance, transform=positive())

        self.trainable_nu = trainable_nu
        if self.trainable_nu and tf.math.is_inf(nu):
            raise ValueError("Cannot have trainable `nu` parameter with infinite value")

        self.nu: Union[float, TensorType, np.ndarray, gpflow.Parameter]
        if self.trainable_nu:
            self.nu = gpflow.Parameter(nu, transform=positive())
        else:
            self.nu = nu

    @property
    def space(self) -> Union[Space, List[Space]]:
        r"""Alias to the `base_kernel`\ s space property."""
        return self.base_kernel.space

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> TensorType:
        """Evaluate the covariance matrix K(X, X2) (or K(X, X) if X2=None)."""
        lengthscale = tf.convert_to_tensor(self.lengthscales)
        nu = tf.cast(tf.convert_to_tensor(self.nu), lengthscale.dtype)
        variance = tf.convert_to_tensor(self.variance)
        params = dict(lengthscale=lengthscale, nu=nu)
        return variance * self.base_kernel.K(params, X, X2)

    def K_diag(self, X: TensorType) -> TensorType:
        """Evaluate the diagonal of the covariance matrix K(X, X)."""
        lengthscale = tf.convert_to_tensor(self.lengthscales)
        nu = tf.cast(tf.convert_to_tensor(self.nu), lengthscale.dtype)
        variance = tf.convert_to_tensor(self.variance)
        params = dict(lengthscale=lengthscale, nu=nu)
        return variance * self.base_kernel.K_diag(params, X)


class DefaultFloatZeroMeanFunction(gpflow.mean_functions.Constant):
    """
    Zero mean function. The default GPflow's `ZeroMeanFunction` uses the
    input's dtype as output type, this minor adaptation uses GPflow's
    `default_float` instead. This is to allow integer-valued inputs, like
    in the case of :class:`~.spaces.Graph` and :class:`~.spaces.Mesh`.
    """

    def __init__(self, output_dim: int = 1):
        super().__init__()
        self.output_dim = output_dim
        del self.c

    def __call__(self, inputs: TensorType) -> TensorType:
        output_shape = tf.concat([tf.shape(inputs)[:-1], [self.output_dim]], axis=0)
        return tf.zeros(output_shape, dtype=gpflow.default_float())
