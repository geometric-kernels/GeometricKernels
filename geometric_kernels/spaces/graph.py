"""
Graphs: TODO
"""
from geometric_kernels.eigenfunctions import Eigenfunctions
from geometric_kernels.spaces import SpaceWithEigenDecomposition
from geometric_kernels.types import TensorLike

# from networkx import Graph


class Graph(SpaceWithEigenDecomposition):
    """TODO"""

    @property
    def dimension(self) -> int:
        pass

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        pass

    def get_eigenvalues(self, num: int) -> TensorLike:
        """
        First `num` eigenvalues of the Laplace-Beltrami operator

        :return: [num, 1] array containing the eigenvalues
        """
        pass
