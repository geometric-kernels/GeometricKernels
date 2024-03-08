"""
GPyTorch kernel wrapper.
"""
from typing import Union

import gpytorch
import lab as B
import torch

from geometric_kernels.kernels import BaseGeometricKernel
from geometric_kernels.spaces.base import Space


class GPyTorchGeometricKernel(gpytorch.kernels.Kernel):
    """
    GPyTorch wrapper for :class:`BaseGeometricKernel`.

    **Note**: the `base_kernel` itself does not store any of its hyperparameters
    (like `lengthscale` and `nu`), therefore you either need to pass them down to
    this wrapper explicitly or use the default values, as provided by the
    `init_params` method of the `base_kernel`.

    :param base_kernel: the kernel to wrap.
    :type base_kernel: :class:`BaseGeometricKernel`
    :param lengthscale: initial value of the length scale. If not given or set
        to None, uses the default value of the `base_kernel`, as provided by
        its `init_params` method.
    :type lengthscale: Union[B.Float, B.TorchNumeric, B.NPNumeric], optional
    :param nu: initial value of the length scale. If not given or set
        to None, uses the default value of the `base_kernel`, as provided by
        its `init_params` method.
    :type nu: Union[B.Float, B.TorchNumeric, B.NPNumeric], optional
    :param variance: initial value of the variance (outputscale) of the kernel,
        defaults to 1.0.
    :type variance: Union[B.Float, B.TorchNumeric, B.NPNumeric], optional
    :param trainable_nu: whether or not the parameter nu is to be optimized
        over. Cannot be True if nu is equal to infinity. You cannot change
        this parameter after constructing the object. Defaults to False.
    :type trainable_nu: bool, optional

    :raises ValueError: if trying to set nu = infinity together with
        trainable_nu = True.
    """

    has_lengthscale = True

    def __init__(
        self,
        base_kernel: BaseGeometricKernel,
        lengthscale: Union[B.Float, B.TorchNumeric, B.NPNumeric] = None,
        nu: Union[B.Float, B.TorchNumeric, B.NPNumeric] = None,
        variance: Union[B.Float, B.TorchNumeric, B.NPNumeric] = 1.0,
        trainable_nu: bool = False,
        **kwargs,
    ):
        """
        Initialize a GPyTorchGeometricKernel object.
        """
        super().__init__(**kwargs)

        self.base_kernel = base_kernel

        default_params = base_kernel.init_params()

        if nu is None:
            nu = default_params["nu"]

        if lengthscale is None:
            lengthscale = default_params["lengthscale"]

        lengthscale = torch.as_tensor(lengthscale)
        variance = torch.as_tensor(variance)
        nu = torch.as_tensor(nu)

        self._trainable_nu = trainable_nu
        if self._trainable_nu and torch.isinf(nu):
            raise ValueError("Cannot have trainable `nu` parameter with infinite value")

        self.lengthscale = lengthscale

        self.register_parameter(
            name="raw_variance", parameter=torch.nn.Parameter(torch.tensor(1.0))
        )
        self.register_constraint("raw_variance", gpytorch.constraints.Positive())
        self.variance = variance

        if self._trainable_nu:
            self.register_parameter(
                name="raw_nu", parameter=torch.nn.Parameter(torch.tensor(1.0))
            )
            self.register_constraint("raw_nu", gpytorch.constraints.Positive())
            self.nu = nu
        else:
            self.register_buffer("raw_nu", nu)

    @property
    def space(self) -> Space:
        """Alias to kernel space"""
        return self.base_kernel.space

    @property
    def variance(self) -> torch.Tensor:
        """The variance parameter"""
        return self.raw_variance_constraint.transform(self.raw_variance)

    @variance.setter
    def variance(self, value):
        value = torch.as_tensor(value).to(self.raw_variance)
        self.initialize(
            raw_variance=self.raw_variance_constraint.inverse_transform(value)
        )

    @property
    def nu(self) -> torch.Tensor:
        """The smoothness parameter"""
        if self._trainable_nu:
            return self.raw_nu_constraint.transform(self.raw_nu)
        else:
            return self.raw_nu

    @nu.setter
    def nu(self, value):
        if self._trainable_nu:
            if torch.isinf(value):
                raise ValueError(
                    "Cannot have infinite `nu` value when trainable_nu = True"
                )
            value = torch.as_tensor(value).to(self.raw_nu)
            self.initialize(raw_nu=self.raw_nu_constraint.inverse_transform(value))
        else:
            self.raw_nu = torch.as_tensor(value)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **kwargs):
        """Evaluate kernel"""
        params = dict(lengthscale=self.lengthscale, nu=self.nu)
        if diag:
            return self.base_kernel.K_diag(params, x1)
        return self.variance * self.base_kernel.K(params, x1, x2)
