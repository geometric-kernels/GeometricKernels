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
    """

    def __init__(self, *kernels):
        self.kernels = kernels

    dimensions = [kernel.space.dimension for kernel in self.kernels]
    self.dimension_indices = []
    i = 0
    inds = B.linspace(0, sum(dimensions) - 1, sum(dimensions)).astype(int)
    for dim in dimensions:
        self.dimension_indices.append(inds[i : i + dim])
        i += dim

    @property
    def space(self) -> List[Space]:
        return [kernel.space for kernel in self.kernels]

    def init_params_and_state(self) -> Tuple(List[dict], List[dict]):
        params_and_state = [kernel.init_params_and_state() for kernel in self.kernels]

        return [l[0] for l in params_and_state], [l[1] for l in params_and_state]

    def K(self, params, state, X, X2=None, **kwargs) -> B.Numeric:
        if X2 is None:
            X2 = X

        Xs = [B.take(X, inds, axis=-1) for inds in self.dimension_indices]
        X2s = [B.take(X2, inds, axis=-1) for inds in self.dimension_indices]

        return B.stack(
            *[kernel.K(X, X2, p, s) for X, X2, p, s in zip(Xs, X2s, params, state)],
            axis=-1,
        ).prod(dim=-1)

    def K_diag(self, params, state, X):
        Xs = [B.take(X, inds, axis=-1) for inds in self.dimension_indices]

        return B.stack(
            *[kernel.K_diag(X, p, s) for X, p, s in zip(Xs, params, state)],
            axis=-1,
        ).prod(dim=-1)
