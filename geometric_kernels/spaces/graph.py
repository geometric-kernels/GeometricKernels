"""
This module provides the :class:`Graph` space.
"""

import lab as B
import numpy as np
from beartype.typing import Dict, Tuple

from geometric_kernels.lab_extras import (
    degree,
    dtype_integer,
    eigenpairs,
    reciprocal_no_nan,
    set_value,
)
from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import (
    Eigenfunctions,
    EigenfunctionsFromEigenvectors,
)


class Graph(DiscreteSpectrumSpace):
    """
    The GeometricKernels space representing the node set of any user-provided
    weighted undirected graph.

    The elements of this space are represented by node indices, integer values
    from 0 to n-1, where n is the number of nodes in the user-provided graph.

    Each individual eigenfunction constitutes a *level*.

    .. note::
        A tutorial on how to use this space is available in the
        :doc:`Graph.ipynb </examples/Graph>` notebook.

    :param adjacency_matrix:
        An n-dimensional square, symmetric matrix, where
        adjacency_matrix[i, j] is non-zero if there is an edge
        between nodes i and j. SciPy's sparse matrices are supported.

        .. warning::
            Make sure that the type of the `adjacency_matrix` is of the
            backend (NumPy (or SciPy) / JAX / TensorFlow, PyTorch) that
            you wish to use for internal computations.

    :param normalize_laplacian:
        If True, the graph Laplacian will be degree normalized (symmetrically):
        L_sym = D^-0.5 * L * D^-0.5.

        Defaults to False.

    .. admonition:: Citation

        If you use this GeometricKernels space in your research, please consider
        citing :cite:t:`borovitskiy2021`.
    """

    def __init__(self, adjacency_matrix: B.Numeric, normalize_laplacian: bool = False):  # type: ignore
        self.cache: Dict[int, Tuple[B.Numeric, B.Numeric]] = {}
        self._checks(adjacency_matrix)
        self._set_laplacian(adjacency_matrix, normalize_laplacian)  # type: ignore
        self._normalized = normalize_laplacian

    def __str__(self):
        return f"Graph({self.num_vertices}, {'normalized' if self._normalized else 'unnormalized'})"

    @staticmethod
    def _checks(adjacency):
        """
        Checks if `adjacency` is a square symmetric matrix.
        """
        assert (
            len(adjacency.shape) == 2 and adjacency.shape[0] == adjacency.shape[1]
        ), "Matrix is not square."

        assert not B.any(adjacency != B.T(adjacency)), "Adjacency is not symmetric"

    @property
    def dimension(self) -> int:
        """
        :return:
            0.
        """
        return 0  # this is needed for the kernel math to work out

    @property
    def num_vertices(self) -> int:
        """
        Number of vertices in the graph.
        """
        return self._laplacian.shape[0]

    def _set_laplacian(self, adjacency, normalize_laplacian=False):
        """
        Construct the appropriate graph Laplacian from the adjacency matrix.
        """
        degree_matrix = degree(adjacency)
        self._laplacian = degree_matrix - adjacency
        if normalize_laplacian:
            degree_inv_sqrt = reciprocal_no_nan(B.sqrt(degree_matrix))
            self._laplacian = degree_inv_sqrt @ self._laplacian @ degree_inv_sqrt

    def get_eigensystem(self, num):
        """
        Returns the first `num` eigenvalues and eigenvectors of the graph Laplacian.
        Caches the solution to prevent re-computing the same values.

        .. note::
            If the `adjacency_matrix` was a sparse SciPy array, requesting
            **all** eigenpairs will lead to a conversion of the sparse matrix
            to a dense one due to scipy.sparse.linalg.eigsh limitations.

        :param num:
            Number of eigenpairs to return. Performs the computation at the
            first call. Afterwards, fetches the result from cache.

        :return:
            A tuple of eigenvectors [n, num], eigenvalues [num, 1].
        """
        assert (
            num <= self.num_vertices
        ), "Number of eigenpairs cannot exceed the number of vertices"
        if num not in self.cache:
            evals, evecs = eigenpairs(self._laplacian, num)

            evecs *= B.sqrt(self.num_vertices)

            eps = np.finfo(float).eps
            for i, evalue in enumerate(evals):
                if evalue < eps or evalue < 0:
                    evals = set_value(evals, i, eps)  # lowest eigenvals should be zero

            self.cache[num] = (evecs, evals[:, None])

        return self.cache[num]

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        Returns the :class:`~.EigenfunctionsFromEigenvectors` object with
        `num` levels (i.e., in this case, `num` eigenpairs).

        :param num:
            Number of levels.
        """
        eigenfunctions = EigenfunctionsFromEigenvectors(self.get_eigenvectors(num))
        return eigenfunctions

    def get_eigenvectors(self, num: int) -> B.Numeric:
        """
        :param num:
            Number of eigenvectors to return.

        :return:
            Array of eigenvectors, with shape [n, num].
        """
        return self.get_eigensystem(num)[0]

    def get_eigenvalues(self, num: int) -> B.Numeric:
        """
        :param num:
            Number of eigenvalues to return.

        :return:
            Array of eigenvalues, with shape [num, 1].
        """
        return self.get_eigensystem(num)[1]

    def get_repeated_eigenvalues(self, num: int) -> B.Numeric:
        """
        Same as :meth:`get_eigenvalues`.

        :param num:
            Same as :meth:`get_eigenvalues`.
        """
        return self.get_eigenvalues(num)

    def random(self, key, number):
        num_vertices = B.shape(self._laplacian)[0]
        key, random_vertices = B.randint(
            key, dtype_integer(key), number, 1, lower=0, upper=num_vertices
        )

        return key, random_vertices

    @property
    def element_shape(self):
        """
        :return:
            [1].
        """
        return [1]

    @property
    def element_dtype(self):
        """
        :return:
            B.Int.
        """
        return B.Int
