"""
This module provides the :class:`EdgeGraph` space.
"""

import lab as B
from beartype.typing import Dict, Tuple, Optional

from geometric_kernels.lab_extras import (
    dtype_integer,
    eigenpairs,
)

from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import (
    Eigenfunctions,
    EigenfunctionsFromEigenvectors,
)


class GraphEdge(DiscreteSpectrumSpace):
    """
    The GeometricKernels space representing the edge set of any user-provided undirected graph.

    The elements of this space are represented by edge indices, integer values
    from 0 to m-1, where m is the number of edges in the user-provided graph.

    Each individual eigenfunction constitutes a *level*.

    .. note::
        A tutorial on how to use this space is available in the
        :doc:`EdgeSpaceGraph.ipynb </examples/EdgeSpaceGraph>` notebook.

    :param incidence_matrix:
        An n x m dimensional matrix where n is the number of nodes and m is the number of edges.
        incidence_matrix[i, e] is -1 if node i is the start node of edge e, 1 if node i is the end node of edge e.

        Note that the -1s and 1s in the incidence matrix do not necessarily mean that the graph is directed. Instead, they are used to allow oriented computations along edges.

    :param triangle_incidence_matrix: optional
        An m x t dimensional matrix where m is the number of edges and t is the number of triangles.
        triangle_incidence_matrix[e, t] is -1 if edge e is anti-aligned with triangle t, 1 if edge e is aligned with triangle t, 0 otherwise.

    .. note:
        The Laplacian is an m x m dimensional matrix where m is the number of
        edges which is computed as
            incidence_matrix.T @ incidence_matrix, if triangle_incidence_matrix is None
            incidence_matrix.T @ incidence_matrix + triangle_incidence_matrix @ triangle_incidence_matrix.T, if triangle_incidence_matrix is not None

    .. admonition:: Citation

        If you use this GeometricKernels space in your research, please consider
        citing :cite:t:`yang2024`.
    """

    def __init__(
        self, incidence_matrix: B.Int, triangle_incidence_matrix: Optional[B.Int] = None
    ):
        self.cache: Dict[int, Tuple[B.Numeric, B.Numeric]] = {}
        self._checks(incidence_matrix, triangle_incidence_matrix)

        self.num_vertices, self.num_edges = incidence_matrix.shape
        if triangle_incidence_matrix is None:
            self.num_triangles = 0
        else:
            self.num_triangles = triangle_incidence_matrix.shape[1]

        self._set_laplacian(incidence_matrix, triangle_incidence_matrix)

    @staticmethod
    def _checks(incidence_matrix, triangle_incidence_matrix):
        """
        Checks if `incidence_matrix` and `triangle_incidence_matrix` are of
        appropriate structure.
        """

        assert B.rank(incidence_matrix) == 2, "Incidence matrix must be a 2-dim tensor."

        assert B.all(
            B.sum(incidence_matrix == 1, axis=0) == 1
        ), "Each column of the incidence matrix must contain 1 exactly once."
        assert B.all(
            B.sum(incidence_matrix == -1, axis=0) == 1
        ), "Each column of the incidence matrix must contain -1 exactly once."

        if triangle_incidence_matrix is not None:
            assert (
                B.rank(triangle_incidence_matrix) == 2
            ), "Triangle incidence matrix matrix must be a 2-dim tensor."

            num_edges = B.shape(incidence_matrix)[1]
            assert B.shape(triangle_incidence_matrix)[0] == num_edges, (
                "The first dimension of the triangle incidence matrix must"
                " be equal to the second dimension of the incidence matrix."
            )

            assert B.all(B.sum(triangle_incidence_matrix != 0, axis=0) == 3), (
                "Each column of the edge triangle incidence matrix must"
                " have exactly 3 non-zero entries."
            )

    @property
    def dimension(self) -> int:
        """
        :return:
            0.
        """
        return 0  # this is needed for the kernel math to work out

    def _set_laplacian(self, incidence_matrix, triangle_incidence_matrix):
        """
        Construct the appropriate graph Laplacian from the adjacency matrix.
        """
        self._down_edge_laplacian = incidence_matrix.T @ incidence_matrix

        if triangle_incidence_matrix is None:
            self._up_edge_laplacian = B.zeros(
                B.dtype(self._down_edge_laplacian), *B.shape(self._down_edge_laplacian)
            )
        else:
            self._up_edge_laplacian = B.matmul(
                triangle_incidence_matrix, triangle_incidence_matrix, tr_b=True
            )

        self._edge_laplacian = self._down_edge_laplacian + self._up_edge_laplacian

    def get_eigensystem(self, num, hgc_order=False):
        """
        Returns the first `num` eigenvalues and eigenvectors of the Hodge Laplacian.
        Caches the solution to prevent re-computing the same values.

        :param num:
            Number of eigenpairs to return. Performs the computation at the
            first call. Afterwards, fetches the result from cache.

        :param hgc_order:
            If True, the eigenmodes are organized according to the Harmonic, Gradient and Curl eigenvalues from small to big.


        :return:
            A tuple of eigenvectors [m, num], eigenvalues [num, 1].
        """
        eps = 1e-6
        if num not in self.cache:
            evals, evecs = eigenpairs(self._edge_laplacian, num)
            if self.num_triangles > 0 and hgc_order:
                # harmonic ones are the ones associated to zero eigenvalues of the edge laplacian
                total_var = []
                total_div = []
                total_curl = []
                num_eigemodes = len(evals)
                for i in range(num_eigemodes):
                    total_var.append(
                        B.matmul(
                            evecs[:, i].reshape(1, -1),
                            B.matmul(self._edge_laplacian, evecs[:, i]),
                        )
                    )
                    total_div.append(
                        B.matmul(
                            evecs[:, i].reshape(1, -1),
                            B.matmul(self._down_edge_laplacian, evecs[:, i]),
                        )
                    )
                    total_curl.append(
                        B.matmul(
                            evecs[:, i].reshape(1, -1),
                            B.matmul(self._up_edge_laplacian, evecs[:, i]),
                        )
                    )

                harm_evecs, grad_evecs, curl_evecs = [], [], []
                for i in range(num_eigemodes):
                    if total_var[i] < eps:
                        harm_evecs.append(i)
                    elif total_div[i] > eps:
                        grad_evecs.append(i)
                    elif total_curl[i] > eps:
                        curl_evecs.append(i)

                reorder_indices = harm_evecs + grad_evecs + curl_evecs
                assert (
                    len(reorder_indices) == num_eigemodes
                ), "The eigenmodes are not correctly organized"

                evals = B.take(evals, reorder_indices, axis=0)
                evecs = B.take(evecs, reorder_indices, axis=1)

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
            Array of eigenvectors, with shape [m, num].
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
        key, random_edges = B.randint(
            key, dtype_integer(key), number, 1, lower=0, upper=self.num_edges
        )

        return key, random_edges

    @property
    def element_shape(self):
        """
        :return:
            [1].
        """
        return [1]
