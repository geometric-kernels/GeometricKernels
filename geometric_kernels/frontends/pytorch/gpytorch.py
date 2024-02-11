"""
GPyTorch kernel wrapper
"""
import gpytorch
import torch

from geometric_kernels.kernels import BaseGeometricKernel
from geometric_kernels.spaces.base import Space


class GPyTorchGeometricKernel(gpytorch.kernels.Kernel):
    """
    GPyTorch wrapper for `BaseGeometricKernel`
    """

    has_lengthscale = True

    def __init__(
        self,
        base_kernel: BaseGeometricKernel,
        lengthscale=None,
        nu=None,
        variance=1.0,
        trainable_nu: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.base_kernel = base_kernel

        default_params = base_kernel.init_params()

        if nu is None:
            nu = default_params["nu"]

        if lengthscale is None:
            lengthscale = default_params["lengthscale"]

        self.lengthscale = torch.tensor(lengthscale)

        self.trainable_nu = trainable_nu
        if self.trainable_nu and torch.isinf(nu):
            raise ValueError("Cannot have trainable `nu` parameter with infinite value")

        self.register_parameter(
            name="raw_variance", parameter=torch.nn.Parameter(torch.tensor(1.0))
        )
        self.register_constraint("raw_variance", gpytorch.constraints.Positive())
        self.variance = variance

        if self.trainable_nu:
            self.register_parameter(
                name="raw_nu", parameter=torch.nn.Parameter(torch.tensor(1.0))
            )
            self.register_constraint("raw_nu", gpytorch.constraints.Positive())
            self.nu = nu
        else:
            self.register_buffer("raw_nu", torch.tensor(nu))

    @property
    def space(self) -> Space:
        """Alias to kernel Space"""
        return self.base_kernel.space

    @property
    def variance(self) -> torch.Tensor:
        """The variance parameter"""
        return self.raw_variance_constraint.transform(self.raw_variance)

    @variance.setter
    def variance(self, value):
        value = torch.as_tensor(value).to(self.raw_variance)
        self.initialize(raw_variance=self.raw_variance_constraint.inverse_transform(value))

    @property
    def nu(self) -> torch.Tensor:
        """The smoothness parameter"""
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
            return self.base_kernel.K_diag(params, x1)
        return self.variance*self.base_kernel.K(params, x1, x2)
