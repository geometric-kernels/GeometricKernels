"""
Pytorch kernel wrapper
"""
import gpytorch
import numpy as np
import torch

from geometric_kernels.kernels import BaseGeometricKernel
from geometric_kernels.spaces.base import Space


class GPytorchGeometricKernel(gpytorch.kernels.Kernel):
    """
    Pytorch wrapper for `BaseGeometricKernel`
    """

    has_lengthscale = True

    def __init__(
        self,
        kernel: BaseGeometricKernel,
        lengthscale=1.0,
        nu=np.inf,
        trainable_nu: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._kernel = kernel

        params, state = self._kernel.init_params_and_state()

        self.state = state

        self.lengthscale = torch.tensor(lengthscale)

        self.trainable_nu = trainable_nu
        if self.trainable_nu and torch.isinf(nu):
            raise ValueError("Cannot have trainable `nu` parameter with infinite value")

        if self.tranable_nu:
            self.register_parameter(
                name="raw_nu", parameter=torch.nn.Parameter(torch.tensor(nu))
            )
            self.register_constraint("raw_nu", gpytorch.constraints.Positive())
        else:
            self.register_buffer("raw_nu", torch.tensor(nu))

    @property
    def space(self) -> Space:
        """Alias to kernel Space"""
        return self._kernel.space

    @property
    def nu(self) -> torch.Tensor:
        """A smoothness parameter"""
        if self.trainable_nu:
            return self.raw_nu_constraint.transform(self.raw_nu)
        else:
            return self.raw_nu

    @nu.setter
    def nu(self, value):
        if self.trainable_nu:
            value = torch.as_tensor(value).to(self.raw_nu)
            self.initialize(raw_nu=self.raw_nu_constraint.inverse_transform(value))
        else:
            self.raw_nu = torch.as_tensor(value)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **kwargs):
        """Eval kernel"""
        # TODO: check batching dimensions

        params = dict(lengthscale=self.lengthscale, nu=self.nu)
        if diag:
            return self._kernel.K_diag(params, self.state, x1)
        return self._kernel.K(params, self.state, x1, x2)
