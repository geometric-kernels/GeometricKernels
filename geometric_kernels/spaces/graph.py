"""
Graph object
"""

import warnings
from typing import List

import lab as B
import numpy as np
from scipy.sparse.linalg import eigsh
# from networkx import Graph

from geometric_kernels.eigenfunctions import Eigenfunctions
from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.mesh import ConvertEigenvectorsToEigenfunctions


class Graph(DiscreteSpectrumSpace):
    """
    Represents an arbitrary undirected graph.
    """

    def __init__(self, adjacency_matrix: List[np.array]):
        """
        :param adjacency_matrix: An n-dimensional square matrix
            representing edges.
        """
        adjacency_matrix = self._checks(adjacency_matrix)
        self.cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.set_laplacian(adjacency_matrix.astype('float'))

    @staticmethod
    def _checks(adjacency):
        # check dtype
        assert isinstance(adjacency, np.ndarray) and \
               len(adjacency.shape) == 2 and \
               adjacency.shape[0] == adjacency.shape[1]

        if not np.isin(adjacency, [0, 1]).all():
            warnings.warn("Adjacency is weighted, ignoring weights.")
            adjacency = self._discard_edge_weights(adjacency)

        if not (adjacency == adjacency.T).all():
            warnings.warn("Adjacency is not symmetric, ignoring directions.")
            adjacency = self._discard_edge_weights(adjacency + adjacency.T)

        return adjacency

    @staticmethod
    def _discard_edge_weights(adjacency):
        adjacency[adjacency > 0] = 1
        return adjacency

    @property
    def dimension(self) -> int:
        """ TODO(AR): Make sure this is correct. """
        return 0

    def set_laplacian(self, adjacency):
        self._laplacian = np.diag(adjacency.sum(axis=0)) - \
                          adjacency

    def get_eigensystem(self, num):
        """
        Returns the first `num` eigenvalues and eigenvectors of the graph Laplacian.
        Caches the solution to prevent re-computing the same values.

        TODO(AR): Make sure this is optimal.

        :param num: number of eigenvalues and functions to return.
        :return: A Tuple of eigenvectors [n, num], eigenvalues [num, 1]
        """
        if num not in self.cache:
            evals, evecs = eigsh(self._laplacian, num, sigma=1e-8)
            # evecs, _ = np.linalg.qr(evecs)
            self.cache[num] = (evecs, evals.reshape(-1, 1))

        return self.cache[num]

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        First `num` eigenfunctions of the Laplace-Beltrami operator on the Graph.

        :param num: number of eigenfunctions returned
        :return: eigenfu [n, num]
        """
        eigenfunctions = ConvertEigenvectorsToEigenfunctions(self.get_eigenvectors(num))
        return eigenfunctions

    def get_eigenvectors(self, num: int) -> B.Numeric:
        """
        :param num: number of eigenvectors returned
        :return: eigenvectors [n, num]
        """
        return self.get_eigensystem(num)[0]

    def get_eigenvalues(self, num: int) -> B.Numeric:
        """
        :param num: number of eigenvalues returned
        :return: eigenvalues [num, 1]
        """
        return self.get_eigensystem(num)[1]

