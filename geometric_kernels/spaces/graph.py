"""
Graph object
"""

from typing import Dict, Tuple

import lab as B
import numpy as np
import scipy.sparse as sp

from geometric_kernels.eigenfunctions import Eigenfunctions
from geometric_kernels.lab_extras import degree, eigenpairs, take_along_axis
from geometric_kernels.spaces.base import DiscreteSpectrumSpace


class ConvertEigenvectorsToEigenfunctions(Eigenfunctions):
    """
    Converts the array of eigenvectors to a callable objects,
    where inputs are given by the indices. Based on
    from geometric_kernels.spaces.mesh import ConvertEigenvectorsToEigenfunctions.
    TODO(AR): Combine this and mesh.ConvertEigenvectorsToEigenfunctions.
    """

    def __init__(self, eigenvectors: B.Numeric):
        """
        :param eigenvectors: [Nv, M]
        """
        self.eigenvectors = eigenvectors

    def __call__(self, X: B.Numeric, **parameters) -> B.Numeric:
        """
        Selects `N` locations from the `M` eigenvectors.

        :param X: indices [N, 1]
        :param parameters: unused
        :return: [N, M]
        """
        indices = B.cast(B.dtype_int(X), X)
        Phi = take_along_axis(self.eigenvectors, indices, axis=0)
        return Phi

    def num_eigenfunctions(self) -> int:
        """Number of eigenvectos, M"""
        return B.shape(self.eigenvectors)[-1]


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
        self.set_laplacian(adjacency_matrix)  # type: ignore

    @property
    def dimension(self) -> int:
        return 0  # this is needed for the kernel math to work out

    def set_laplacian(self, adjacency):
        self._laplacian = degree(adjacency) - adjacency

    def get_eigensystem(self, num):
        """
        Returns the first `num` eigenvalues and eigenvectors of the graph Laplacian.
        Caches the solution to prevent re-computing the same values. Note that, if a
        sparse scipy matrix is input, requesting all n eigenpairs will lead to a
        conversion of the sparse matrix to a dense one due to scipy.sparse.linalg.eigsh
        limitations.

        TODO(AR): Make sure this is optimal.

        :param num: number of eigenvalues and functions to return.
        :return: A Tuple of eigenvectors [n, num], eigenvalues [num, 1]
        """
        if num not in self.cache:
            evals, evecs = eigenpairs(self._laplacian, num)

            if evals[0] < 0:
                evals[0] = np.finfo(float).eps  # lowest eigenval should be zero

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
