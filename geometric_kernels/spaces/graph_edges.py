"""
This module provides the :class:`EdgeGraph` space.
"""

import lab as B
import numpy as np
from beartype.typing import Dict, List, Optional, Tuple, Union
from scipy.sparse import csr_array, lil_array, sparray, spmatrix

from geometric_kernels.lab_extras import dtype_integer, eigenpairs, int_like
from geometric_kernels.spaces.base import HodgeDiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import (
    Eigenfunctions,
    EigenfunctionsFromEigenvectors,
)


class GraphEdges(HodgeDiscreteSpectrumSpace):
    """
    The GeometricKernels space representing the edge set of a user-provided
    graph. The graph must be unweighted, undirected, and without loops.
    However, we assume the graph is oriented: for every edge, either (i, j)
    or (j, i) is chosen as the positively oriented edge, with the other one
    being negatively oriented. Any function f on this space must satisfy
    f(e) = -f(-e) if e is a positively oriented edge and -e is the
    corresponding negatively oriented edge.

    All positively oriented edges are indexed **from 1 to the number of edges
    (inclusive)**. The reason why we start indexing the edges from 1 is to
    allow for edge -e to correspond to the opposite orientation of edge e.
    Importantly, orientation or a particular ordering of edges does not
    affect the geometry of the space. However, changing those requires the
    respective change in the way you represent the data.

    For this space, each individual eigenfunction constitutes a *level*. If
    possible, we recommend using num=n_edges levels. Since the current
    implementation does not support sparse eigensolvers anyway, this should
    not create too much time/memory overhead during the precomputations and
    will also ensure that all types of eigenfunctions are present in the
    eigensystem.

    .. admonition:: Tutorial

        A tutorial on how to use this space is available in the
        :doc:`GraphEdges.ipynb </examples/GraphEdges>` notebook.

    .. admonition:: Theory

        Mathematically, this space is actually a simplicial 2-complex, and
        the functions satisfying the condition f(e) = -f(-e) are called
        1-cochains. These admit a Hodge decomposition into the harmonic,
        gradient, and curl parts. You can find more about Hodge decomposition
        on :doc:`this page </theory/hodge>`.

    .. admonition:: Construction

        **Easy way:** construct the space from the adjacency matrix of the graph.

        The easiest way to construct an instance of this space is to use the
        :meth:`from_adjacency` method, which constructs the space from the
        adjacency matrix of the graph. By default, this would use all possible
        triangles in the graph, but the user can also provide a list of
        triangles explicitly. These triangles define the curl operator on the
        graph and are thus instrumental in defining the Hodge decomposition.
        If you don't know what triangles to provide, you can leave this
        parameter empty, and the space will compute the maximal set of
        triangles for you, which is a good solution in most cases.

        When constructing an instance this way, the orientation of the edges and
        triangles is chosen automatically in such a way that (i, j) is always
        positively oriented if i < j, and the triangles are of form (e1, e2, e3)
        where e1 = (i, j), e2 = (j, k), e3 = -(i, k), and i < j < k.

        **Hard way:** construct the space from the oriented edges and triangles.

        When constructing an instance of this space directly via :meth:`__init__`,
        the user provides an `oriented_edges` array, mapping edge indices to
        ordered pairs of node indices, with order defining the positive
        orientation, and an `oriented_triangles` array, mapping triangle indices
        to triples of signed edge indices. These arrays must be compatible in
        the sense that the edges in triangles must be connected. This way, the
        user has full control over the orientation and ordering of the edges.

    .. admonition:: Complexity
        The current implementation of the GraphEdges space is supposed to occupy
        O(n_edges^2) memory and take O(n_edges^3) time to compute the eigensystem.

        Currently, it does not support sparse eigensolvers, which would allow
        storing the Laplacian matrices in a sparse format and potentially reduce
        the O(n_edges^3) complexity to O(n_edges) with a large constant.

    :param n_nodes:
        The number of nodes in the graph.

    :param oriented_edges:
        A 2D array of shape [n_edges, 2], where n_edges is the number of edges
        in the graph. Each row of the array represents an oriented edge, i.e.,
        a pair of node indices. If `oriented_edges[e]` is `[i, j]`, then the
        edge with index e+1 (edge indexing starts from 1) represents an edge
        from node `i` to node `j` which is considered positively oriented, while
        the edge with index -e-1 represents an edge from node `j` to node `i`.

        Make sure that `oriented_edges` and `oriented_triangles` are of the
        backend (NumPy / JAX / TensorFlow, PyTorch) that you wish to use for
        internal computations.

    :param oriented_triangles:
        A 2D array of shape [n_triangles, 3], where n_triangles is the number
        of triangles in the graph. Each row of the array represents an
        oriented triangle, i.e., a triple of signed edge indices. If
        `oriented_triangles[t]` is `[i, j, k]`, then `t` consists of the edges
        `|i|`, `|j|`, and `|k|`, where the sign of the edge index indicates
        the orientation of the edge. Optional. If not provided or set to
        `None`, we assume that the set of triangles is empty.

        Make sure that `oriented_edges` and `oriented_triangles` are of the
        backend (NumPy / JAX / TensorFlow, PyTorch) that you wish to use for
        internal computations.

    :param checks_mode:
        A keyword argument that determines the level of checks performed on
        the oriented_edges and oriented_triangles arrays. The default value is
        "simple", which performs only basic checks. The other possible values
        are "comprehensive" and "skip", which perform more extensive checks
        and no checks, respectively. The "comprehensive" mode is useful for
        debugging but can be slow for large graphs.

    :param index:
        Optional. A sparse matrix of shape [n_nodes, n_nodes] such that
        `index[i, j]` is the index of the edge `(i, j)` in the `oriented_edges`.
        `index[i, j]` is positive if (i, j) corresponds to positive orientation
        of the edge, and negative if it corresponds to the negative orientation.
        If provided, should be compatible with the `oriented_edges` array.

    .. admonition:: Citation

        If you use this GeometricKernels space in your research, please consider
        citing :cite:t:`yang2024`.
    """

    def __init__(
        self,
        n_nodes: int,
        oriented_edges: B.Int,
        oriented_triangles: Optional[B.Int],
        *,
        checks_mode: str = "simple",
        index: Optional[csr_array] = None,
    ):
        if checks_mode not in ["simple", "comprehensive", "skip"]:
            raise ValueError(
                "The `checks_mode` parameter must be 'simple', 'comprehensive', or 'skip'."
            )
        else:
            do_checks = checks_mode != "skip"
            comprehensive_checks = checks_mode == "comprehensive"

        self.cache: Dict[int, Tuple[B.Numeric, B.Numeric]] = {}
        self.n_nodes = n_nodes

        if do_checks:
            self._checks_oriented_edges(oriented_edges, n_nodes, comprehensive_checks)
        self.n_edges = oriented_edges.shape[0]
        self.oriented_edges = oriented_edges

        if oriented_triangles is not None:
            if do_checks:
                self._checks_oriented_triangles(
                    oriented_triangles, comprehensive_checks
                )
            if comprehensive_checks:
                self._checks_compatible(oriented_triangles)
            self.n_triangles = oriented_triangles.shape[0]
        else:
            self.n_triangles = 0
        self.oriented_triangles = oriented_triangles

        if index is not None:
            if comprehensive_checks:
                self._check_index(index)
            self.index = index
        else:
            self.index = self.compute_index(n_nodes, oriented_edges)

        self._set_laplacian()

    @staticmethod
    def compute_index(n_nodes: int, oriented_edges: B.Numeric) -> csr_array:
        """
        Construct the index matrix from the oriented edges.

        :param n_nodes:
            The number of nodes in the graph.

        :param oriented_edges:
            The oriented edges array.

        :return:
            The index matrix. A scipy csr_array of shape [n_nodes, n_nodes] such
            that `index[i, j]` is the index of the edge `(i, j)`.
        """
        result = lil_array((n_nodes, n_nodes), dtype=int)
        for i in range(oriented_edges.shape[0]):
            result[oriented_edges[i, 0], oriented_edges[i, 1]] = i + 1
            result[oriented_edges[i, 1], oriented_edges[i, 0]] = -i - 1
        return result.tocsr()

    def _checks_oriented_edges(
        self, oriented_edges: B.Numeric, n_nodes: int, comprehensive: bool = False
    ):
        """
        Checks if `oriented_edges` is of appropriate structure.

        :param oriented_edges:
            The oriented edges array.

        :param comprehensive:
            If True, perform more extensive checks.
        """

        assert (
            B.rank(oriented_edges) == 2
        ), "The oriented_edges array must be 2-dimensional."

        assert B.shape(oriented_edges)[1] == 2, "oriented_edges must have shape (*, 2)."

        assert B.dtype(oriented_edges) == int_like(
            oriented_edges
        ), "The oriented_edges must be an array of integers."
        assert B.all(
            oriented_edges >= 0
        ), "The oriented_edges array must contain only non-negative values."
        assert B.all(
            oriented_edges < self.n_nodes
        ), "The values in the oriented_edges array must be < self.n_nodes."
        assert B.all(
            oriented_edges[:, 0] - oriented_edges[:, 1] != 0
        ), "Loops are not allowed."

        if comprehensive:
            n_edges = oriented_edges.shape[0]

            for i in range(n_edges):
                for j in range(i + 1, n_edges):
                    assert B.any(
                        oriented_edges[i, :] != oriented_edges[j, :]
                    ), "The oriented_edges array must not contain duplicate edges."
                    assert B.any(
                        oriented_edges[i, :] != oriented_edges[j, ::-1]
                    ), "The oriented_edges array must not contain duplicate edges."

            assert set(range(self.n_nodes)) == set(
                B.to_numpy(B.flatten(oriented_edges))
            ), "The oriented_edges array must contain all nodes."

    def _checks_oriented_triangles(
        self, oriented_triangles: B.Numeric, comprehensive=False
    ):
        """
        Checks if `oriented_triangles` is of appropriate structure.

        :param oriented_triangles:
            The oriented triangles array.

        :param comprehensive:
            If True, perform more extensive checks.
        """

        assert (
            B.rank(oriented_triangles) == 2
        ), "The oriented_triangles array must be 2-dimensional."

        assert (
            B.shape(oriented_triangles)[1] == 3
        ), "oriented_triangles must have shape (*, 3)."

        assert B.dtype(oriented_triangles) == int_like(
            oriented_triangles
        ), "The oriented_triangles must be an array of integers."
        assert B.all(
            B.abs(oriented_triangles) >= 1
        ), "The oriented_triangles array must contain only non-zero values."
        assert B.all(
            B.abs(oriented_triangles) <= self.n_edges
        ), "The absolute values in the oriented_triangles array must be <= self.n_edges."

        assert B.all(
            B.abs(oriented_triangles) < self.n_edges
        ), "The absolute values in the oriented_triangles array must be less than self.n_edges."
        assert (
            B.all(
                B.abs(oriented_triangles[:, 0]) - B.abs(oriented_triangles[:, 1]) != 0
            )
            or B.all(
                B.abs(oriented_triangles[:, 0]) - B.abs(oriented_triangles[:, 2]) != 0
            )
            or B.all(
                B.abs(oriented_triangles[:, 1]) - B.abs(oriented_triangles[:, 2]) != 0
            )
        ), "Triangles must consist of 3 different edges."

        if comprehensive:
            n_triangles = oriented_triangles.shape[0]

            for i in range(n_triangles):
                for j in range(i + 1, n_triangles):
                    assert B.any(
                        oriented_triangles[i, :] != oriented_triangles[j, :]
                    ), "The oriented_triangles array must not contain duplicate triangles."

    def _checks_compatible(
        self,
        oriented_triangles: B.Numeric,
    ):
        """
        Checks if `self.oriented_edges` and `oriented_triangles` are compatible.

        :param oriented_triangles:
            The oriented triangles array.
        """

        assert B.dtype(self.oriented_edges) == B.dtype(
            oriented_triangles
        ), "The oriented_edges and oriented_triangles arrays must have the same dtype."

        n_triangles = oriented_triangles.shape[0]
        for t in range(n_triangles):
            resolved_edges = self.resolve_edges(oriented_triangles[t, :])
            assert (
                resolved_edges[0, 1] == resolved_edges[1, 0]
            ), "The edges in the triangle must be connected."
            assert (
                resolved_edges[1, 1] == resolved_edges[2, 0]
            ), "The edges in the triangle must be connected."
            assert (
                resolved_edges[2, 1] == resolved_edges[0, 0]
            ), "The edges in the triangle must be connected."

    def _check_index(self, index: csr_array):
        edges = []
        for e in range(1, self.oriented_edges.shape[0] + 1):
            i, j = self.oriented_edges[e - 1, :]
            assert (
                index[i, j] == e
            ), "The index matrix must be compatible with oriented_edges."
            assert (
                index[j, i] == -e
            ), "The index matrix must be compatible with oriented_edges."
            edges.append((min(i, j), max(i, j)))

        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if (i, j) not in edges:
                    assert (
                        index[i, j] == 0
                    ), "The index matrix must be compatible with oriented_edges."
                    assert (
                        index[j, i] == 0
                    ), "The index matrix must be compatible with oriented_edges."

    def resolve_edges(self, es: B.Int) -> B.Int:
        """
        Resolve the signed edge indices to node indices.

        :param es:
            A 1-dimensional array of edge indices. Each edge index is a number
            from 1 to the number of edges (incl.).

        :return:
            A 2-dimensional array `result` such that `result[e, :]` is `[i, j]`
            where |e| = (i, j) if e > 0 and |e| = (j, i) if e < 0.
        """
        assert B.rank(es) == 1
        assert B.all(B.abs(es) >= 1) and B.all(B.abs(es) <= self.n_edges)

        result = self.oriented_edges[B.abs(es) - 1]
        result = B.where(B.expand_dims(es > 0, axis=-1), result, result[:, ::-1])
        return result

    def resolve_triangles(self, ts: B.Int) -> B.Int:
        """
        Resolve the triangle indices to node indices.

        :param ts:
            A 1-dimensional array of triangle indices. Each triangle index is a
            number from 0 to the number of triangles.

        :return:
            A 3-dimensional array `result` such that `result[t, :]` is `[i, j, k]`
            where i = e1[0], j = e2[0], k = e3[0], and e1, e2, e3 are the
            oriented edges constituting the triangle `t`.
        """
        assert B.rank(ts) == 1
        assert B.all(B.abs(ts) >= 0) and B.all(B.abs(ts) < self.n_triangles)

        edge_indices = B.flatten(
            self.oriented_triangles[ts]
        )  # [N,] -> [N, 3] -> [N*3,]
        edges = B.reshape(
            self.resolve_edges(edge_indices), len(ts), 3, 2
        )  # [N*3,] -> [N*3, 2] -> [N, 3, 2]
        return edges[:, :, 0]  # [N, 3, 2] -> [N, 3]

    @classmethod
    def from_adjacency(
        cls,
        adjacency_matrix: Union[B.NPNumeric, sparray, spmatrix],
        type_reference: B.RandomState,
        *,
        triangles: Optional[List[Tuple[int, int, int]]] = None,
        checks_mode: str = "simple",
    ) -> "GraphEdges":
        """
        Construct the GraphEdges space from the adjacency matrix of a graph.

        :param adjacency_matrix:
            Adjacency matrix of a graph. A numpy array of shape
            `[n_nodes, n_nodes]` where `n_nodes` is the number of nodes in the
            graph. `adjacency_matrix[i, j]` can only be 0 or 1.

        :param type_reference:
            A random state object of the preferred backend to infer backend from.

        :param triangles:
            A list of tuples of three integers representing the nodes of the
            triangles in the graph. If not provided or set to None, the maximal
            possible set of triangles will be computed and used.

        :param checks_mode:
            Forwards the `checks_mode` parameter to the constructor.

        :return:
            A constructed instance of the GraphEdges space.
        """
        if isinstance(adjacency_matrix, np.ndarray):
            index = csr_array(adjacency_matrix, dtype=int)
        elif isinstance(adjacency_matrix, (sparray, spmatrix)):
            index = csr_array(adjacency_matrix, dtype=int, copy=True)

        if len(index.shape) != 2:
            raise ValueError("Adjacency matrix must be a square matrix.")
        if (abs(index - index.T) > 1e-10).nnz != 0:
            raise ValueError("Adjacency matrix must be symmetric.")
        if (index.diagonal() != 0).any():
            raise ValueError("Adjacency matrix must have zeros on the diagonal.")
        if np.sum(index.data == 1) + np.sum(index.data == 0) != len(index.data):
            raise ValueError("Adjacency matrix can only contain zeros and ones.")

        number_of_nodes = index.shape[0]
        number_of_edges = np.sum(index.data) // 2

        oriented_edges = B.zeros(dtype_integer(type_reference), number_of_edges, 2)

        cur_edge_ind = 1
        for i in range(number_of_nodes):
            for j in range(i + 1, number_of_nodes):
                if index[i, j] == 1:
                    oriented_edges[cur_edge_ind - 1, 0] = i
                    oriented_edges[cur_edge_ind - 1, 1] = j
                    # We also store cur_edge_ind in the index matrix for later use
                    index[i, j] = cur_edge_ind
                    index[j, i] = -index[i, j]
                    cur_edge_ind += 1
        assert (
            cur_edge_ind == number_of_edges + 1
        )  # double check that we have the right number of edges

        if triangles is None:
            triangles = []
            for i in range(number_of_nodes):
                for j in range(i + 1, number_of_nodes):
                    for k in range(j + 1, number_of_nodes):
                        if index[i, j] != 0 and index[j, k] != 0 and index[i, k] != 0:
                            triangles.append((i, j, k))

        number_of_triangles = len(triangles)
        oriented_triangles = B.zeros(
            dtype_integer(type_reference), number_of_triangles, 3
        )
        for triangle_index, triangle in enumerate(triangles):
            i, j, k = sorted(triangle)  # sort the nodes in the triangle
            oriented_triangles[triangle_index, 0] = index[i, j]
            oriented_triangles[triangle_index, 1] = index[j, k]
            oriented_triangles[triangle_index, 2] = index[k, i]

        return cls(
            number_of_nodes,
            oriented_edges,
            oriented_triangles,
            checks_mode=checks_mode,
            index=index,
        )

    @property
    def dimension(self) -> int:
        """
        :return:
            0.
        """
        return 0  # this is needed for the kernel math to work out

    def _set_laplacian(self):
        """
        Construct the appropriate graph Laplacian from the adjacency matrix.
        """

        # This does node_to_edge_incidence.T @ node_to_edge_incidence
        # TODO: make this more efficient, avoid for loops.
        self._down_laplacian = B.zeros(
            B.dtype(self.oriented_edges), self.n_edges, self.n_edges
        )
        for i in range(self.n_edges):
            for j in range(self.n_edges):
                self._down_laplacian[i, j] = (
                    int(self.oriented_edges[i, 0] == self.oriented_edges[j, 0])
                    + int(self.oriented_edges[i, 1] == self.oriented_edges[j, 1])
                    - int(self.oriented_edges[i, 0] == self.oriented_edges[j, 1])
                    - int(self.oriented_edges[i, 1] == self.oriented_edges[j, 0])
                )

        # node_to_edge_incidence = B.zeros(
        #     B.dtype(self.oriented_edges), self.n_nodes, self.n_edges
        # )
        # for i in range(self.n_edges):
        #     node_to_edge_incidence[self.oriented_edges[i, 0], i] = -1
        #     node_to_edge_incidence[self.oriented_edges[i, 1], i] = 1
        # down_laplacian_alt = B.matmul(node_to_edge_incidence, node_to_edge_incidence, tr_a=True)
        # assert B.all(down_laplacian_alt == self._down_laplacian)

        # This does edge_to_triangle_incidence @ edge_to_triangle_incidence.T
        # TODO: make this more efficient, avoid for loops.
        self._up_laplacian = B.zeros(
            B.dtype(self.oriented_edges), self.n_edges, self.n_edges
        )
        for i in range(self.n_edges):
            for j in range(self.n_edges):
                for t in range(self.n_triangles):
                    cur_triangle = set(B.to_numpy(self.oriented_triangles[t, :]))
                    if (i + 1 in cur_triangle and j + 1 in cur_triangle) or (
                        -i - 1 in cur_triangle and -j - 1 in cur_triangle
                    ):
                        self._up_laplacian[i, j] += 1
                    if (i + 1 in cur_triangle and -j - 1 in cur_triangle) or (
                        -i - 1 in cur_triangle and j + 1 in cur_triangle
                    ):
                        self._up_laplacian[i, j] += -1

        # edge_to_triangle_incidence = B.zeros(
        #     B.dtype(self.oriented_triangles), self.n_edges, self.n_triangles
        # )
        # for i in range(self.n_triangles):
        #     edge_indices = B.abs(self.oriented_triangles[i, :])
        #     signs = self.oriented_triangles[i, :] / edge_indices
        #     edge_to_triangle_incidence[edge_indices[0] - 1, i] = signs[0]
        #     edge_to_triangle_incidence[edge_indices[1] - 1, i] = signs[1]
        #     edge_to_triangle_incidence[edge_indices[2] - 1, i] = signs[2]
        # up_laplacian_alt = B.matmul(edge_to_triangle_incidence, edge_to_triangle_incidence, tr_b=True)
        # assert B.all(up_laplacian_alt == self._up_laplacian)

        self._hodge_laplacian = self._down_laplacian + self._up_laplacian

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

        # The current implementation computes the complete set of eigenpairs of
        # three n_edges x n_edges Laplacian matrices. Since we don't currently
        # support sparse Laplacians, this should not be a major bottleneck.
        #
        # Here are some important points for anyone who wants to add proper
        # support for sparse Laplacians and sparse eigensolvers which allow
        # computing only a few of the most important eigenpairs, like
        # scipy.sparse.linalg.eigsh.
        #
        # 1. You cannot just compute the eigenpairs of the Hodge Laplacian and
        #    then filter them into the harmonic, gradient, or curl parts. This
        #    is because up and down edge Laplacians may have common non-zero
        #    eigenvalues. This means that the eigenspaces of the Hodge
        #    Laplacian that correspond to such eigenvalues will contain both
        #    gradient and curl types of eigenvectors. When an eigenspace is
        #    multi-dimensional, an eigensolver can return an arbitrary basis
        #    for it. Thus, for example, you may get a basis where all the vectors
        #    have non-zero grad and curl components at the same time. Thus, you
        #    won't be able to filter them.
        #
        #    If you put in more effort, you can take the basis for each non-zero
        #    eigenvalue with multiplicity > 1, project each vector onto the
        #    pure-gradient and pure-curl subspaces (using up and down Laplacians),
        #    and then orthogonalize the projected vectors.
        #
        # 2. You cannot just request `num` eigenpairs from the Hodge Laplacian,
        #    up Laplacian and down Laplacian. Imagine that you have a space
        #    with 5 harmonic, 100 gradient, and 100 curl eigenpairs. Then,
        #    the first 105 eigenpairs of the up Laplacian will correspond to
        #    harmonic and gradient eigenvectors, both corresponding to zero
        #    eigenvalues, while only the last 100 eigenpairs will correspond
        #    to the actual curl eigenvectors that we are interested in getting
        #    from the up Laplacian. The same is true for the down Laplacian.
        #
        # 3. An elegant solution would be to construct
        #    > diff_laplacian = up_laplacian - down_laplacian
        #    and run a sparse eigensolver on this matrix, requesting `num`
        #    eigenpairs with the lowest absolute value of eigenvalues. Then,
        #    you get eigenpairs of the Hodge Laplacian corresponding to the
        #    first `num` smallest eigenvalues, but the gradient and curl
        #    eigenvectors would correspond to eigenvalues of different signs
        #    and thus would be separated.

        eps = 1e-6

        if num not in self.cache:
            # We use Hodge Laplacian to find the eigenpairs of harmonic type,
            # these are the ones associated to zero eigenvalues.
            evals_hodge, evecs_hodge = eigenpairs(self._hodge_laplacian, self.n_edges)

            # We use up and down Laplacians to find the eigenpairs of curl and
            # gradient types. These are the ones associated to non-zero eigenvalues.
            evals_up, evecs_up = eigenpairs(self._up_laplacian, self.n_edges)
            evals_down, evecs_down = eigenpairs(self._down_laplacian, self.n_edges)

            # We count the number of eigenpairs of each type for future reference.
            n_harm = B.sum(evals_hodge < eps)
            n_curl, n_grad = B.sum(evals_up >= eps), B.sum(evals_down >= eps)

            # We concatenate the eigenvalues and eigenvectors of harmonic, curl, and
            # gradient types into a single array.
            evals = B.concat(
                evals_hodge[evals_hodge < eps],
                evals_up[evals_up >= eps],
                evals_down[evals_down >= eps],
            )
            evecs = B.concat(
                evecs_hodge[:, evals_hodge < eps],
                evecs_up[:, evals_up >= eps],
                evecs_down[:, evals_down >= eps],
                axis=1,
            )

            evecs *= B.sqrt(self.n_edges)  # to make sure average variance is 1.

            # We get the indices that would sort the eigenvalues in ascending order.
            sorted_indices = B.argsort(evals)

            # In harmonic_idx, gradient_idx, and curl_idx, we store the indices of
            # the eigenvalues and eigenvectors of harmonic, gradient, and curl types.
            # We also make sure that the indices are not greater than `num`.
            def filter_inds(inds: B.Int, num: int) -> List[int]:
                return [i for i in inds if i < num]

            harmonic_idx = filter_inds(sorted_indices[:n_harm], num)
            curl_idx = filter_inds(sorted_indices[n_harm : n_harm + n_curl], num)
            grad_idx = filter_inds(
                sorted_indices[n_harm + n_curl : n_harm + n_curl + n_grad], num
            )

            # We sort the eigenvalues and eigenvectors in ascending order of
            # eigenvalues, keeping only `num` of the first eigenpairs.
            sorted_indices = sorted_indices[:num]
            evals = evals[sorted_indices]
            evecs = evecs[:, sorted_indices]

            self.cache[num] = {
                "evals": evals,
                "evecs": evecs,
                "harmonic_idx": harmonic_idx,
                "gradient_idx": curl_idx,
                "curl_idx": grad_idx,
            }

        return self.cache[num]

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
        eigenfunctions = EigenfunctionsFromEigenvectors(
            eigensystem["evecs"][:, idx], index_from_one=True
        )
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

        :param hodge_type:
            The type of the eigenbasis. It can be 'harmonic', 'gradient', or 'curl'.

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
        key, random_edges = B.randint(
            key,
            dtype_integer(key),
            number,
            1,
            lower=1,
            upper=self.n_edges + 1,
        )

        return key, random_edges

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
