"""
Pytorch kernel wrapper
"""
import gpytorch
import torch

from geometric_kernels.kernels import BaseGeometricKernel
from geometric_kernels.spaces.base import Space


class GPytorchGeometricKernel(gpytorch.kernels.Kernel):
    """
    Pytorch wrapper for `BaseGeometricKernel`
    """

    has_lengthscale = True

    def __init__(self, kernel: BaseGeometricKernel, **kwargs):
        super().__init__(**kwargs)

        self._kernel = kernel

        params = self._kernel.init_params()

        self.lengthscale = torch.tensor(params["lengthscale"])
        self.register_parameter(
            name="raw_nu", parameter=torch.nn.Parameter(torch.tensor(params["nu"]))
        )
        self.register_constraint("raw_nu", gpytorch.constraints.Positive())

    @property
    def space(self) -> Space:
        """Alias to kernel Space"""
        return self._kernel.space

    @property
    def nu(self) -> torch.Tensor:
        """A smoothness parameter"""
        return self.raw_nu_constraint.transform(self.raw_nu)

    @nu.setter
    def nu(self, value):
        value = torch.as_tensor(value).to(self.raw_nu)
        self.initialize(raw_nu=self.raw_nu_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **kwargs):
        """Eval kernel"""
        # TODO: check batching dimensions

        params = dict(lengthscale=self.lengthscale, nu=self.nu)
        if diag:
            return self._kernel.K_diag(params, x1)
        return self._kernel.K(params, x1, x2)
