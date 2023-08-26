"""
Graph object
"""

from typing import Dict, Tuple

import lab as B
import numpy as np

from geometric_kernels.lab_extras import (
    degree,
    dtype_integer,
    eigenpairs,
    reciprocal_no_nan,
    set_value,
)
from geometric_kernels.spaces.base import (
    ConvertEigenvectorsToEigenfunctions,
    DiscreteSpectrumSpace,
)
from geometric_kernels.spaces.eigenfunctions import Eigenfunctions


class Graph(DiscreteSpectrumSpace):
    """
    Represents an arbitrary undirected graph.
    """

    def __init__(self, adjacency_matrix: B.Numeric, normalize_laplacian: bool = False):  # type: ignore
        """
        :param adjacency_matrix: An n-dimensional square, symmetric matrix,
            where adjacency_matrix[i, j] is non-zero if there is an edge
            between nodes i and j. Scipy's sparse matrices are supported.
        :param normalize_laplacian: If True, the Laplacian will be degree
            normalized (symmetrically). L_sym = D^-0.5 * L * D^-0.5
        """
        self.cache: Dict[int, Tuple[B.Numeric, B.Numeric]] = {}
        self._checks(adjacency_matrix)
        self.set_laplacian(adjacency_matrix, normalize_laplacian)  # type: ignore

    @staticmethod
    def _checks(adjacency):
        assert (
            len(B.shape(adjacency)) == 2 and adjacency.shape[0] == adjacency.shape[1]
        ), "Matrix is not square."

        # this is more efficient than (adj == adj.T).all()
        assert not B.any(adjacency != B.T(adjacency)), "Adjacency is not symmetric."

    @property
    def dimension(self) -> int:
        return 0  # this is needed for the kernel math to work out

    @property
    def num_vertices(self) -> int:
        return self._laplacian.shape[0]

    def set_laplacian(self, adjacency, normalize_laplacian=False):
        degree_matrix = degree(adjacency)
        self._laplacian = degree_matrix - adjacency
        if normalize_laplacian:
            degree_inv_sqrt = reciprocal_no_nan(B.sqrt(degree_matrix))
            self._laplacian = degree_inv_sqrt @ self._laplacian @ degree_inv_sqrt

    def get_eigensystem(self, num):
        """
        Returns the first `num` eigenvalues and eigenvectors of the graph Laplacian.
        Caches the solution to prevent re-computing the same values. Note that, if a
        sparse scipy matrix is input, requesting all n eigenpairs will lead to a
        conversion of the sparse matrix to a dense one due to scipy.sparse.linalg.eigsh
        limitations.

        :param num: number of eigenvalues and functions to return.
        :return: A Tuple of eigenvectors [n, num], eigenvalues [num, 1]
        """
        if num not in self.cache:
            evals, evecs = eigenpairs(self._laplacian, num)

            evecs *= np.sqrt(self.num_vertices)

            eps = np.finfo(float).eps
            for i, evalue in enumerate(evals):
                if evalue < eps:
                    evals = set_value(evals, i, eps)  # lowest eigenvals should be zero

            self.cache[num] = (evecs, evals[:, None])

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

    def get_repeated_eigenvalues(self, num: int) -> B.Numeric:
        """
        :param num: number of eigenvalues
        :return: eigenvalues [num, 1]
        """
        return self.get_eigenvalues(num)

    def random(self, key, number):
        num_vertices = B.shape(self._laplacian)[0]
        key, random_vertices = B.randint(
            key, dtype_integer(key), number, 1, lower=0, upper=num_vertices
        )

        return key, random_vertices
