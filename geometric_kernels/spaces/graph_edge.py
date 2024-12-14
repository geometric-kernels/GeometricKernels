"""
This module provides the :class:`EdgeGraph` space.
"""

import lab as B
import numpy as np
from beartype.typing import Dict, List, Optional, Tuple, Union
from scipy.sparse import csr_array, sparray, spmatrix

from geometric_kernels.lab_extras import dtype_integer, eigenpairs, int_like
from geometric_kernels.spaces.base import HodgeDiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import (
    Eigenfunctions,
    EigenfunctionsFromEigenvectors,
)


class GraphEdge(HodgeDiscreteSpectrumSpace):
    """
    The GeometricKernels space representing the node set of any user-provided
    weighted undirected, but **oriented** graph without loops.
    Actually, this space is a simplicial 2-complex, as explained below.

    The elements of this space are indices of positively oriented edges of the graph.
    Edge indices are integers from 1 to the number of edges in the graph (inclusive).
    The reason for indexing the edges from 1 is to allow for signed values.

    The fact that the graph is oriented means that despite the edge (j, i) is
    always present in the graph as long as the edge (i, j) is present, any
    function f on the set of edges must satisfy f(i, j) = -f(j, i). Mathematically,
    this space is actually a simplicial 2-complex, and the functions satisfying
    the above condition are called 1-cochains.

    The easiest way to construct an instance of this space is to use the
    :meth:`from_adjacency` method, which constructs the space from the adjacency
    matrix of the graph. By default this would use all possible triangles in the
    graph, but the user can also provide a list of triangles explicitly. These
    triangles define the curl operator on the graph and are thus instrumental in
    defining the Hodge decomposition. If you don't know what triangles to provide,
    you can leave this parameter empty, and the space will compute the maximal
    set of triangles for you, which is arguably a good solution in most cases.
    When constructing an instance this way, the orientation of the edges and triangles
    is chosen automatically in such a way that (i, j) is always oriented from i to j
    if i < j and the triangles are of form (e1, e2, e3) where e1 = (i, j),
    e2 = (j, k), and e3 = -(i, k).

    When constructing an instance of this space directly, the user chooses the
    orientation by providing an `oriented_edges` array, mapping edge indices
    to pairs of node indices. And an `oriented_triangles` array, mapping triangle
    indices to triples of signed edge indices. These arrays must be compatible in
    the sense that the edges in the triangles must be connected.

    For this space, each individual eigenfunction constitutes a *level*. If
    possible, we recommend using num=nun_edges levels. Since the current
    implementation does not support sparse eigensolvers anyway, this should
    not create too much time/memory overhead during the precomputations, and will
    also ensure that all types of eigenfunctions are present in the eigensystem.

    .. note::
        A tutorial on how to use this space is available in the
        :doc:`GraphEdge.ipynb </examples/GraphEdge>` notebook.

    .. warning::
        Make sure that `oriented_edges` and `oriented_triangles` are of the
        backend (NumPy / JAX / TensorFlow, PyTorch) that you wish to use for
        internal computations.

    :param num_nodes:
        The number of nodes in the graph.

    :param oriented_edges:
        A 2D array of shape [n_edges, 2], where n_edges is the number of edges
        in the graph. Each row of the array represents an oriented edge, i.e.
        a pair of node indices. If `oriented_edges[e]` is `[i, j]`, then edge `e`
        is oriented from node `i` to node `j`.

    :param oriented_triangles:
        A 2D array of shape [n_triangles, 3], where n_triangles is the number of
        triangles in the graph. Each row of the array represents an oriented
        triangle, i.e. a triple of signed edge indices. If `oriented_triangles[t]`
        is `[i, j, k]`, then `t` consists of the edges `|i|`, `|j|`, and `|k|`,
        where the sign of the edge index indicates the orientation of the edge.

        Optional. If not provided or set to `None`, we assume that the set of
        triangles is empty.

    :param checks_mode:
        A keyword argument that determines the level of checks performed on the
        oriented_edges and oriented_triangles arrays. The default value is "simple"
        which performs only basic checks. The other possible value is "comprehensive"
        and "skip" which perform more extensive checks and no checks, respectively.
        The "comprehensive" mode is useful for debugging, but can be slow for large
        graphs.

    :param index:
        Optional. A sparse matrix of shape [num_nodes, num_nodes] such that
        `index[i, j]` is the index of the edge `(i, j)` in the `oriented_edges`,
        if (i, j) is positively oriented, and minus index of the edge `(i, j)`
        otherwise. If not provided, will be constructed automatically.
        Should be compatible with the `oriented_edges` array.

    .. admonition:: Citation
        If you use this GeometricKernels space in your research, please consider
        citing :cite:t:`yang2024`.
    """

    def __init__(
        self,
        num_nodes: int,
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
        self.num_nodes = num_nodes

        if do_checks:
            self._checks_oriented_edges(oriented_edges, num_nodes, comprehensive_checks)
        self.num_edges = oriented_edges.shape[0]
        self.oriented_edges = oriented_edges

        if oriented_triangles is not None:
            if do_checks:
                self._checks_oriented_triangles(
                    oriented_triangles, self.num_edges, comprehensive_checks
                )
            if comprehensive_checks:
                self._checks_compatible(oriented_edges, oriented_triangles, num_nodes)
            self.num_triangles = oriented_triangles.shape[0]
        else:
            self.num_triangles = 0
        self.oriented_triangles = oriented_triangles

        if index is not None:
            self._check_index(index)
            self.index = index
        else:
            self.index = compute_index(oriented_edges)

        self._set_laplacian()

    def _checks_oriented_edges(
        self, oriented_edges: B.Numeric, num_nodes: int, comprehensive: bool = False
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
            oriented_edges < self.num_nodes
        ), "The values in the oriented_edges array must be < self.num_nodes."
        assert B.all(
            oriented_edges[:, 0] - oriented_edges[:, 1] != 0
        ), "Loops are not allowed."

        if comprehensive:
            num_edges = oriented_edges.shape[0]

            for i in range(num_edges):
                for j in range(i + 1, num_edges):
                    assert B.any(
                        oriented_edges[i, :] != oriented_edges[j, :]
                    ), "The oriented_edges array must not contain duplicate edges."

            assert set(range(self.num_nodes)) == set(
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
            oriented_triangles >= 1
        ), "The oriented_triangles array must contain only strictly positive values."
        assert B.all(
            oriented_triangles <= self.num_edges
        ), "The values in the oriented_triangles array must be <= self.num_edges."

        assert B.all(
            B.abs(oriented_triangles) < self.num_edges
        ), "The absolute values in the oriented_edges array must be less than self.num_edges."
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
        ), "Triangle must consist of 3 _different_ edges."

        if comprehensive:
            num_triangles = oriented_triangles.shape[0]

            for i in range(num_triangles):
                for j in range(i + 1, num_triangles):
                    assert B.any(
                        num_triangles[i, :] != num_triangles[j, :]
                    ), "The num_triangles array must not contain duplicate triangles."

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

        num_triangles = oriented_triangles.shape[0]
        for t in range(num_triangles):
            resolved_edges = self.resolve_edge(oriented_triangles[t, :])
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
        TODO

    def resolve_edge(self, e: B.Int) -> B.Int:
        """
        Resolve the signed edge indices to node indices.

        :param e:
            A 1-dimensional array of edge indices. Each edge index is a number
            from 1 to the number of edges (incl.).

        :return:
            A 2-dimensional array `result` such that `result[e, :]` is `[i, j]`
            where |e| = (i, j) if e > 0 and |e| = (j, i) if e < 0.
        """
        assert B.rank(e) == 1
        assert B.all(B.abs(e) >= 1) and B.all(B.abs(e) <= self.num_edges)

        result = self.edges[B.abs(e) - 1]
        # TODO: will not work with TF/JAX
        result[e < 0] = result[e < 0][::-1]

        return result

    @classmethod
    def from_adjacency(
        cls,
        adjacency_matrix: Union[B.NPNumeric, sparray, spmatrix],
        type_reference: B.RandomState,
        triangles: Optional[List[Tuple[int, int, int]]] = None,
        *,
        checks_mode: str = "simple",
    ) -> "GraphEdge":
        """
        Construct the GraphEdge space from the adjacency matrix of the graph.

        :param adjacency_matrix:
            Adjacency matrix of the graph. A numpy array of shape [n_nodes, n_nodes]
            where n_nodes is the number of nodes in the graph. adjacency_matrix[i, j]
            can only be 0 or 1.

        :param type_reference:
            A random state object of the preferred backend to infer backend from.

        :param triangles:
            A list of tuples of three integers representing the nodes of the
            triangles in the graph. If not provided or set to None, the maximal
            possible set of triangles will be computed and used.

        :param checks_mode:
            Forwards the `checks_mode` parameter to the constructor.

        :return:
            A constructed instance of the GraphEdge space.
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

        cur_edge_ind = 0
        for i in range(number_of_nodes):
            for j in range(i + 1, number_of_nodes):
                if index[i, j] == 1:
                    oriented_edges[cur_edge_ind, 0] = i
                    oriented_edges[cur_edge_ind, 1] = j
                    # We also store cur_edge_ind in the index matrix for later use
                    index[i, j] = cur_edge_ind + 1
                    index[j, i] = -index[i, j]
                    cur_edge_ind += 1
        assert (
            cur_edge_ind == number_of_edges
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

        # TODO: make this more efficient, avoid for loops.
        self._down_laplacian = B.zeros(
            B.dtype(self.oriented_edges), self.num_edges, self.num_edges
        )
        for i in range(self.num_edges):
            for j in range(self.num_edges):
                self._down_laplacian[i, j] = (
                    self.oriented_edges[i, 0]
                    == self.oriented_edges[j, 0] + self.oriented_edges[i, 1]
                    == self.oriented_edges[j, 1] - self.oriented_edges[i, 0]
                    == self.oriented_edges[j, 1] - self.oriented_edges[i, 1]
                    == self.oriented_edges[j, 0]
                )

        # TODO: make this more efficient, avoid for loops.
        self._up_laplacian = B.zeros(
            B.dtype(self.oriented_edges), self.num_edges, self.num_edges
        )
        for i in range(self.num_edges):
            for j in range(self.num_edges):
                for t in range(self.num_triangles):
                    cur_triangle = set(B.to_numpy(self.oriented_triangles[t, 0]))
                    if (i in cur_triangle and j in cur_triangle) or (
                        -i in cur_triangle and -j in cur_triangle
                    ):
                        self._up_laplacian[i, j] += 1
                    if (i in cur_triangle and -j in cur_triangle) or (
                        -i in cur_triangle and j in cur_triangle
                    ):
                        self._up_laplacian[i, j] += -1

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
        # three num_edges x num_edges matrices. Since we don't currently support
        # sparse laplacian matrices, in this class, this is not a major issue.
        #
        # Here is some important points for anyone who wants to add proper
        # support for sparse eigensolvers which allow to compute only a subset
        # of the most important eigenpairs, like scipy.sparse.linalg.eigsh.
        #
        # 1. You cannot just compute the eigenpairs of the Hodge Laplacian and
        #    then filter them into the harmonic, gradient, or curl parts. This
        #    is because up and down edge Laplacians may have common non-zero
        #    eigenvalues. This means that the eigenspaces of the the Hodge
        #    Laplacian that correspond to such eigenvalues will contain both
        #    gradient and curl type of eigenvectors. When an eigenspace is
        #    multi-dimensional, an eigensolver can return an arbitrary basis
        #    for it. For example, you may get a basis where all the vectors
        #    have non-zero grad and curl components at the same time. Thus, you
        #    won't be able to filter them.
        #
        #    If you put in more effort, you can take the basis for each non-zero
        #    eigenvalue with multiplicity > 1, project each vector onto the
        #    pure-gradient and pure-curl subspaces, using up and down Laplacians,
        #    and then orthogonalize the projected vectors.
        #
        #  2. You cannot just request `num` eigenpairs from the Hodge Laplacian,
        #     up and down Laplacians separately. Imagine that you have a space
        #     with 5 harmonic, 100 gradient, and 100 curl eigenpairs. Then,
        #     first 105 eigenpairs of the up Laplacian will correspond to
        #     harmonic and gradient eigenvectors, both correspond to zero
        #     eigenvalues, while only the last 100 eigenpairs will correspond
        #     to the actual curl eigenvectors that we are interested to get
        #     from the up Laplacian. The same is true for the down Laplacian.
        #
        #  3. An elegant solution would be to construct
        #     > diff_laplacian = up_laplacian - down_laplacian
        #     and run a sparse eigensolver on this matrix, requesting `num`
        #     eigenpairs with the lowest absolute value of eigenvalues. Then,
        #     you get eigenpairs of the Hodge Laplacian corresponding to the
        #     first `num` smallest eigenvalues, but the gradient and curl
        #     eigenvectors would correspond to eigenvalues of different signs.

        eps = 1e-6

        if num not in self.cache:
            # We use Hodge Laplacian to find the eigenpairs of harmonic type, these
            # are the ones associated to zero eigenvalues.
            evals_hodge, evecs_hodge = eigenpairs(self._hodge_laplacian, self.num_edges)

            # We use up and down Laplacians to find the eigenpairs of curl and
            # gradient types. These are the ones associated to non-zero eigenvalues.
            evals_up, evecs_up = eigenpairs(self._up_laplacian, self.num_edges)
            evals_down, evecs_down = eigenpairs(self._down_laplacian, self.num_edges)

            # We count the number of eigenpairs of each type for future reference.
            n_harm = B.sum(evals_hodge < eps)
            n_curl, n_grad = B.sum(evals_up >= eps), B.sum(evals_down >= eps)

            # We concatenate the eigenvalues and eigenvectors of harmonic, curl, and
            # gradient types into a single array.
            evals = B.concatenate(
                evals_hodge[evals_hodge < eps],
                evals_up[evals_up >= eps],
                evals_down[evals_down >= eps],
            )
            evecs = B.concatenate(
                evecs_hodge[:, evals_hodge < eps],
                evecs_up[:, evals_up >= eps],
                evecs_down[:, evals_down >= eps],
                axis=1,
            )

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
        key, random_edges = B.randint(
            key,
            dtype_integer(key),
            number,
            1,
            lower=1,
            upper=self.num_edges + 1,
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
