"""
Graph object
"""

import warnings
from typing import Dict, Tuple

import lab as B
import numpy as np
import scipy.sparse as sp

from geometric_kernels.eigenfunctions import Eigenfunctions
from geometric_kernels.lab_extras import degree
from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.mesh import ConvertEigenvectorsToEigenfunctions

SP_TO_DENSE_WARN = "Converting graph to dense as sp.linalg.eigsh fails if \
num == n in the case of a sparse graph."


class Graph(DiscreteSpectrumSpace):
    """
    Represents an arbitrary undirected graph.
    """

    def __init__(self, adjacency_matrix: Tuple[np.array, sp.spmatrix]):  # type: ignore
        """
        :param adjacency_matrix: An n-dimensional square, symmetric, binary
            matrix, where adjacency_matrix[i, j] is one if there is an edge
            between nodes i and j.
        """
        self.cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.set_laplacian(adjacency_matrix.astype("float"))  # type: ignore

    @property
    def dimension(self) -> int:
        return 0  # this is needed for the kernel math to work out

    def set_laplacian(self, adjacency):
        self._laplacian = degree(adjacency) - adjacency

    def get_eigensystem(self, num):
        """
        Returns the first `num` eigenvalues and eigenvectors of the graph Laplacian.
        Caches the solution to prevent re-computing the same values.

        TODO(AR): Make sure this is optimal.

        :param num: number of eigenvalues and functions to return.
        :return: A Tuple of eigenvectors [n, num], eigenvalues [num, 1]
        """
        if num not in self.cache:
            is_sparse = sp.issparse(self._laplacian)
            all_eigens = num == self._laplacian.shape[0]
            if is_sparse and all_eigens:
                warnings.warn(SP_TO_DENSE_WARN)
                laplacian = self._laplacian.toarray()
            else:
                laplacian = self._laplacian

            evals, evecs = sp.linalg.eigsh(laplacian, num, sigma=1e-8)

            if evals[0] < 0:
                evals[0] = np.finfo(float).eps  # lowest eigenval is frequently -1e-15

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
