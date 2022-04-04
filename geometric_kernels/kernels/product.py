"""
Product of kernels
"""
from typing import List, Tuple

import lab as B

from geometric_kernels.spaces import Space
from geometric_kernels.kernels import BaseGeometricKernel


class ProductGeometricKernel(BaseGeometricKernel):
    """
    Basic implementation of product kernels.
    TODO: Support combined RFF/KL expansion methods

    Parameters
    ----------
    kernels : BaseGeometricKernel
        kernels to compute the product for
    dimension_indices : List[B.Numeric], optional
        List of indices the corespond to the indices of the input that should be fed to each kernel.
        If None, assume the each kernel takes kernel.space.dimension inputs, and that the input will
        be a stack of this size, by default None
    """

    def __init__(
        self,
        *kernels: BaseGeometricKernel,
        dimension_indices: List[B.Numeric] = None,
    ):
        self.kernels = kernels

        if dimension_indices is None:
            dimensions = [kernel.space.dimension for kernel in self.kernels]
            self.dimension_indices = []
            i = 0
            inds = B.linspace(0, sum(dimensions) - 1, sum(dimensions)).astype(int)
            for dim in dimensions:
                self.dimension_indices.append(inds[i : i + dim])
                i += dim
        else:
            assert len(dimension_indices) == len(self.kernels)
            for i in dimension_indices:
                assert i.dtype == B.Int
                assert (i >= 0).all()

            self.dimension_indices = dimension_indices

    @property
    def space(self) -> List[Space]:
        return [kernel.space for kernel in self.kernels]

    def init_params_and_state(self) -> Tuple(List[dict], List[dict]):
        params_and_state = [kernel.init_params_and_state() for kernel in self.kernels]

        return [p[0] for p in params_and_state], [s[1] for s in params_and_state]

    def K(self, params, state, X, X2=None, **kwargs) -> B.Numeric:
        if X2 is None:
            X2 = X

        Xs = [B.take(X, inds, axis=-1) for inds in self.dimension_indices]
        X2s = [B.take(X2, inds, axis=-1) for inds in self.dimension_indices]

        return B.stack(
            *[
                kernel.K(X, X2, p, s)
                for kernel, X, X2, p, s in zip(self.kernels, Xs, X2s, params, state)
            ],
            axis=-1,
        ).prod(dim=-1)

    def K_diag(self, params, state, X):
        Xs = [B.take(X, inds, axis=-1) for inds in self.dimension_indices]

        return B.stack(
            *[
                kernel.K_diag(X, p, s)
                for kernel, X, p, s in zip(self.kernels, Xs, params, state)
            ],
            axis=-1,
        ).prod(dim=-1)
