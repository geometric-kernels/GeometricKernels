"""
This module provides the :class:`EdgeGraph` space.
"""

import lab as B
import networkx as nx
import numpy as np
from beartype.typing import Dict, Optional, Tuple

from geometric_kernels.lab_extras import dtype_integer, eigenpairs
from geometric_kernels.spaces.base import HodgeDiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import (
    Eigenfunctions,
    EigenfunctionsFromEigenvectors,
)


class GraphEdge(HodgeDiscreteSpectrumSpace):
    """
    The GeometricKernels space representing the edge set of any user-provided undirected graph.

    The elements of this space are represented by edge indices, integer values
    from 0 to m-1, where m is the number of edges in the user-provided graph.

    Each individual eigenfunction constitutes a *level*.

    .. note::
        A tutorial on how to use this space is available in the
        :doc:`EdgeSpaceGraph.ipynb </examples/EdgeSpaceGraph>` notebook.

    :param G:
        A networkx graph object.

    :param triangle_list:
        A list of triangles in the graph. If not provided, all triangles in the graph are considered as 2-simplices.

    :param sc_lifting:
        If True, we lift the graph to a simplicial 2-complex using either the provided triangle list or all the triangles in the graph. This acts also a flag to compute the triangle incidence matrix. Defaults to False.

    .. admonition:: Citation

        If you use this GeometricKernels space in your research, please consider
        citing :cite:t:`yang2024`.
    """

    def __init__(
        self,
        node_to_edge_incidence: B.Int,
        edge_to_triangle_incidence: Optional[B.Int] = None,
    ):
        self.cache: Dict[int, Tuple[B.Numeric, B.Numeric]] = {}
        self.num_nodes, self.num_edges = node_to_edge_incidence.shape
        self._checks(node_to_edge_incidence, edge_to_triangle_incidence)
        self._make_oriented(node_to_edge_incidence, edge_to_triangle_incidence)
        self.num_triangles = self.edge_to_triangle_incidence.shape[1]
        self._set_laplacian()

    @staticmethod
    def _checks(node_to_edge_incidence, edge_to_triangle_incidence=None):
        """
        Checks if `node_to_edge_incidence` and `edge_to_triangle_incidence` are of
        appropriate structure.
        """
        assert (
            B.rank(node_to_edge_incidence) == 2
        ), "Incidence matrix must be a 2-dim tensor."
        assert B.all(
            B.sum(node_to_edge_incidence != 0, axis=0) == 2
        ), "Each column of the incidence matrix must contain 2 exactly once."
        if edge_to_triangle_incidence is not None:
            assert (
                B.rank(edge_to_triangle_incidence) == 2
            ), "Triangle incidence matrix matrix must be a 2-dim tensor."
            num_edges = B.shape(node_to_edge_incidence)[1]
            assert B.shape(edge_to_triangle_incidence)[0] == num_edges, (
                "The first dimension of the triangle incidence matrix must"
                " be equal to the second dimension of the incidence matrix."
            )
            assert B.all(B.sum(edge_to_triangle_incidence != 0, axis=0) == 3), (
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

    def _make_oriented(self, node_to_edge_incidence, edge_to_triangle_incidence=None):
        """
        Since our inputs are non-oriented, we make them oriented by taking the vertex with the smallest index first.
        """
        # make the edges oriented
        self.edges = []
        self.nodes = list(range(1, self.num_nodes + 1))

        for i in range(node_to_edge_incidence.shape[1]):
            self.edges.append(tuple(np.where(node_to_edge_incidence[:, i] != 0)[0] + 1))

        self.node_to_edge_incidence = B.zeros(self.num_nodes, self.num_edges)
        for i, edge in enumerate(self.edges):
            assert (
                edge[0] < edge[1]
            )  # "The first node should have a smaller index than the second node."
            self.node_to_edge_incidence[edge[0] - 1, i] = -1
            self.node_to_edge_incidence[edge[1] - 1, i] = 1

        # make the triangles oriented
        if edge_to_triangle_incidence is None:
            cliques = nx.enumerate_all_cliques(nx.Graph(self.edges))
            triangle_vertices = [x for x in cliques if len(x) == 3]
            # sort the triangles
            self.triangles = [tuple(sorted(tri)) for tri in triangle_vertices]
        else:
            self.triangles = []
            for i in range(edge_to_triangle_incidence.shape[1]):
                edge = self.edges[i]
                edges_of_triangle = np.where(edge_to_triangle_incidence[:, i] != 0)[0]
                nodes_of_triangle = np.unique(
                    np.where(self.node_to_edge_incidence[:, edges_of_triangle] != 0)[0]
                )
                self.triangles.append(tuple(nodes_of_triangle + 1))

        self.edge_to_triangle_incidence = self.build_edge_to_triangle_incidence()

    def _set_laplacian(self):
        """
        Construct the appropriate graph Laplacian from the adjacency matrix.
        """
        self._down_edge_laplacian = (
            self.node_to_edge_incidence.T @ self.node_to_edge_incidence
        )
        self._up_edge_laplacian = B.matmul(
            self.edge_to_triangle_incidence, self.edge_to_triangle_incidence, tr_b=True
        )
        self.hodge_laplacian = self._down_edge_laplacian + self._up_edge_laplacian

    @property
    def incidence_matrices(self):
        """
        Return the incidence matrix
        """
        return self.node_to_edge_incidence, self.edge_to_triangle_incidence

    def get_edge_index(self, edges):
        """
        Get the indices of some provided edges in the edge list.

        Args:
            edges (list): Edges.

        Returns:
            list: Indices of the edges.
        """
        assert isinstance(edges, list)  # "The edges should be a list."
        # each edge should be in the edge list
        assert all(edge in self.edges for edge in edges)
        return B.to_numpy([self.edges.index(edge) for edge in edges])

    def build_edge_to_triangle_incidence(self) -> np.ndarray:
        """
        Create the B2 matrix (edge-triangle) from the triangles.

        The `edge_to_triangle_incidence` parameter is a `numpy` array of shape `(n_edges, n_triangles)` where `n_triangles` is the number of triangles in the graph.

        The entry `edge_to_triangle_incidence[e, t]` is
        - `-1` if edge `e` is anti-aligned with triangle `t`, e.g., $(1,3)$ is anti-aligned with $(1,2,3)$,
        - `1` if edge `e` is aligned with triangle `t`, e.g., $(1,2)$ is aligned with $(1,2,3)$, and
        - `0` otherwise.

        Args:
            triangles (list): List of triangles.
            edges (list): List of edges.

        Returns:
            np.ndarray: B2 matrix.
        """

        triangles = self.triangles
        B2 = np.zeros((len(self.edges), len(triangles)))
        for j, triangle in enumerate(triangles):
            a, b, c = triangle
            try:
                index_a = self.edges.index((a, b))
            except ValueError:
                index_a = self.edges.index((b, a))
            try:
                index_b = self.edges.index((b, c))
            except ValueError:
                index_b = self.edges.index((c, b))
            try:
                index_c = self.edges.index((a, c))
            except ValueError:
                index_c = self.edges.index((c, a))

            B2[index_a, j] = 1
            B2[index_c, j] = -1
            B2[index_b, j] = 1

        return B2

    def get_eigensystem(self, num):
        """
        Returns the first `num` eigenvalues and eigenvectors of the Hodge Laplacian.
        Caches the solution to prevent re-computing the same values.

        :param num:
            Number of eigenpairs to return. Performs the computation at the
            first call. Afterwards, fetches the result from cache.

        :return:
            A tuple of eigenvectors [n, num], eigenvalues [num, 1].
        """
        eps = 1e-6
        if num not in self.cache:
            evals, evecs = eigenpairs(self.hodge_laplacian, num)
            # make a dictionary of the eigenvalues and eigenvectors where the keys are the indices of the eigenvalues, and add another value which indicates the type of the eigenbasis (harmonic, gradient, curl)
            self.hodge_eigenbasis = {}
            # add keys to the dictionary
            # harmonic ones are the ones associated to zero eigenvalues of the edge laplacian
            total_var, total_div, total_curl = [], [], []
            num_eigemodes = len(evals)
            for i in range(num_eigemodes):
                total_var.append(
                    B.matmul(
                        evecs[:, i].reshape(1, -1),
                        B.matmul(self.hodge_laplacian, evecs[:, i]),
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
            harmonic_inds, gradient_inds, curl_inds = [], [], []
            for i in range(num_eigemodes):
                if total_var[i] < eps:
                    harmonic_inds.append(i)
                elif total_div[i] > eps:
                    gradient_inds.append(i)
                elif total_curl[i] > eps:
                    curl_inds.append(i)
            self.hodge_eigenbasis = {
                "evals": evals,
                "evecs": evecs,
                "harmonic_idx": harmonic_inds,
                "gradient_idx": gradient_inds,
                "curl_idx": curl_inds,
            }

            self.cache[num] = self.hodge_eigenbasis

        return self.cache[num]

    # get particular type of eigenbasis
    def get_eigenfunctions(
        self, num: int, hodge_type: Optional[str] = None
    ) -> Eigenfunctions:
        """
        Returns the :class:`~.EigenfunctionsFromEigenvectors` object with
        `num` levels (i.e., in this case, `num` eigenpairs) of a particular type.

        :param num:
            Number of levels.
        :param hodge_type:
            The type of the eigenbasis. It can be 'harmonic', 'gradient', or 'curl'.

        :return:
            EigenfunctionsFromEigenvectors object.
        """
        eigensystem = self.get_eigensystem(num)
        idx = (
            eigensystem[f"{hodge_type}_idx"]
            if hodge_type is not None
            else list(range(num))
        )
        eigenfunctions = EigenfunctionsFromEigenvectors(eigensystem["evecs"][:, idx])
        return eigenfunctions

    def get_eigenvalues(self, num: int, hodge_type: Optional[str] = None) -> B.Numeric:
        """
        Eigenvalues of the Laplacian corresponding to the first `num` levels
        (i.e., in this case, `num` eigenpairs). If `type` is specified, returns
        only the eigenvalues corresponding to the eigenfunctions of that type.
        .. warning::
            If `type` is specified, the array can have fewer than `num` elements.
        :param num:
            Number of levels.
        :return:
            (n, 1)-shaped array containing the eigenvalues. n <= num.
        .. note::
            The notion of *levels* is discussed in the documentation of the
            :class:`~.kernels.MaternKarhunenLoeveKernel`.
        """
        eigensystem = self.get_eigensystem(num)
        idx = (
            eigensystem[f"{hodge_type}_idx"]
            if hodge_type is not None
            else list(range(num))
        )
        eigenvalues = eigensystem["evals"][idx]

        return eigenvalues[:, None]

    def get_repeated_eigenvalues(
        self, num: int, hodge_type: Optional[str] = None
    ) -> B.Numeric:
        """
        Same as :meth:`get_eigenvalues`.
        :param num:
            Same as :meth:`get_eigenvalues`.
        """
        return self.get_eigenvalues(num, hodge_type)

    def random(self, key, number):
        num_edges = B.shape(self.hodge_laplacian)[0]
        key, random_edges_idx = B.randint(
            key,
            dtype_integer(key),
            number,
            1,
            lower=0,
            upper=num_edges,
        )

        random_edges = [self.edges[i] for i in random_edges_idx.flatten().tolist()]

        return key, random_edges, random_edges_idx

    @property
    def element_shape(self):
        """
        :return:
            [1].
        """
        return [1]
