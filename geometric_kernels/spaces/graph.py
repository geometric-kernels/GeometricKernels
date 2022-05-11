"""
Graphs: TODO
"""
import lab as B

from geometric_kernels.eigenfunctions import Eigenfunctions
from geometric_kernels.spaces.base import DiscreteSpectrumSpace

# from networkx import Graph


class Graph(DiscreteSpectrumSpace):
    """TODO"""

    @property
    def dimension(self) -> int:
        pass

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        pass

    def get_eigenvalues(self, num: int) -> B.Numeric:
        """
        First `num` eigenvalues of the Laplace-Beltrami operator

        :return: [num, 1] array containing the eigenvalues
        """
