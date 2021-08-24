"""
Graphs: TODO
"""
import numpy as np

from geometric_kernels.spaces import Space

# from networkx import Graph


class Graph(Space):
    """TODO"""

    @property
    def dimension(self) -> int:
        pass

    def get_eigenfunctions(self, num: int):
        pass

    def get_eigenvalues(self, num: int) -> np.ndarray:
        """
        First `num` eigenvalues of the Laplace-Beltrami operator

        :return: [num, 1] array containing the eigenvalues
        """
        pass
