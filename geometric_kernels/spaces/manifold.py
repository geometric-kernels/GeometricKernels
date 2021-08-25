"""
Manifold: TODO
"""
from typing import Callable

import numpy as np

from geometric_kernels.spaces import SpaceWithEigenDecomposition
from geometric_kernels.types import TensorLike


class Manifold(SpaceWithEigenDecomposition):
    """TODO"""

    @property
    def dimension(self) -> int:
        pass

    def get_eigenfunctions(self, num: int) -> Callable[[TensorLike], TensorLike]:
        pass

    def get_eigenvalues(self, num: int) -> TensorLike:
        """
        First `num` eigenvalues of the Laplace-Beltrami operator

        :return: [num, 1] array containing the eigenvalues
        """
        pass


# from geomstats.geometry.manifold import Manifold
