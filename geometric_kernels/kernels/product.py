"""
Product of kernels
"""

import lab as B

from beartype.typing import List, Mapping, Optional
from geometric_kernels.kernels.base import BaseGeometricKernel
from geometric_kernels.spaces import Space


class ProductGeometricKernel(BaseGeometricKernel):
    def __init__(
        self,
        *kernels: BaseGeometricKernel,
        dimension_indices: Optional[List[B.Numeric]] = None,
    ):
        """
        Basic implementation of product kernels.
        TODO: Support combined RFF/KL expansion methods

        :param kernels: kernels to compute the product for
        :param dimension_indices: List of indices the correspond to the indices of the input that should be fed to each kernel.
            If None, assume the each kernel takes kernel.space.dimension inputs, and that the input will
            be a stack of this size, by default None
        """
        self.kernels = kernels

        if dimension_indices is None:
            dimensions = [kernel.space.dimension for kernel in self.kernels]
            self.dimension_indices: List[B.Numeric] = []
            i = 0
            inds = B.linspace(0, sum(dimensions) - 1, sum(dimensions)).astype(int)
            for dim in dimensions:
                self.dimension_indices.append(inds[i : i + dim])
                i += dim
        else:
            assert len(dimension_indices) == len(self.kernels)
            for idx in dimension_indices:
                assert idx.dtype == B.Int
                assert B.all(idx >= 0)

            self.dimension_indices = dimension_indices

    @property
    def space(self) -> List[Space]:
        return [kernel.space for kernel in self.kernels]

    def init_params(self) -> List[Mapping]:
        params = [kernel.init_params() for kernel in self.kernels]
        return params

    def K(self, params: List[Mapping], X, X2=None, **kwargs) -> B.Numeric:
        if X2 is None:
            X2 = X

        Xs = [B.take(X, inds, axis=-1) for inds in self.dimension_indices]
        X2s = [B.take(X2, inds, axis=-1) for inds in self.dimension_indices]

        return B.prod(
            B.stack(
                *[
                    kernel.K(p, X, X2)
                    for kernel, X, X2, p in zip(self.kernels, Xs, X2s, params)
                ],
                axis=-1,
            ),
            axis=-1,
        )

    def K_diag(self, params, X):
        Xs = [B.take(X, inds, axis=-1) for inds in self.dimension_indices]

        return B.prod(
            B.stack(
                *[
                    kernel.K_diag(p, X)
                    for kernel, X, p in zip(self.kernels, Xs, params)
                ],
                axis=-1,
            ),
            axis=-1,
        )
