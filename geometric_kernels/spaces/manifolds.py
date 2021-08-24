"""
Base manifold and concrete implementation of Manifolds.
"""
from typing import Optional

import geomstats as gs
import numpy as np

from geometric_kernels.spaces import Space
from geometric_kernels.types import TensorLike


class Hypersphere(gs.geometry.hypersphere.Hypersphere, Space):
    """
    Extends geomstats' `Hypersphere` with eigenfunctions and values such that
    it can be used as a `Space`.
    """

    @property
    def dimension(self) -> int:
        return self.dim

    def is_tangent(
        self,
        vector: TensorLike,
        base_point: Optional[TensorLike] = None,
        atol: float = gs.geometry.manifold.ATOL,
    ) -> bool:
        """
        Check whether the `vector` is tangent at `base_point`.

        :param vector: shape=[..., dim]
            Vector to evaluate.
        :param base_point: shape=[..., dim]
            Point on the manifold. Defaults to `None`.
        :param atol: float
            Absolute tolerance.
            Optional, default: 1e-6.

        :return: Boolean denoting if vector is a tangent vector at the base point.
        """
        raise NotImplementedError("`is_tangent` is not implemented for `Hypersphere`")

    def get_eigenfunctions(self, num: int):
        return 0

    def get_eigenvalues(self, num: int) -> np.ndarray:
        """
        First `num` eigenvalues of the Laplace-Beltrami operator

        :return: [num, 1] array containing the eigenvalues
        """
        return 1
