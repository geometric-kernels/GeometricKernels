"""
Pytorch kernel wrapper
"""
import gpytorch
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

    @property
    def space(self) -> Space:
        """Alias to kernel Space"""
        return self._kernel.space

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **kwargs):
        """Eval kernel"""
        # TODO: check batching dimensions
        if diag:
            return self._kernel.K_diag(x1, lengthscale=self.lengthscale)
        return self._kernel.K(x1, x2, lengthscale=self.lengthscale)
