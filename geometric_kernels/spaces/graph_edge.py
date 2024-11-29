"""
This module provides the :class:`Graph` space.
"""

import lab as B
import numpy as np
from beartype.typing import Dict, Tuple, Optional

from geometric_kernels.lab_extras import (
    dtype_integer,
)
from geometric_kernels.spaces.base import HodgeDiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import (
    Eigenfunctions,
    EigenfunctionsFromEigenvectors,
)


class GraphEdge(HodgeDiscreteSpectrumSpace):
    """
    The GeometricKernels space representing the node set of any user-provided
    weighted undirected graph.

    The elements of this space are represented by oriented edges, i.e. 2d arrays
    of node indices, integer values from 0 to n-1, where n is the number of
    nodes in the user-provided graph.

    Each individual eigenfunction constitutes a *level*.

    .. note::
        A tutorial on how to use this space is available in the
        :doc:`GraphEdge.ipynb </examples/GraphEdge>` notebook.

    .. warning::
        Make sure that `node_to_edge_incidence` and `edge_to_triangle_incidence`
        are of the backend (NumPy (or SciPy) / JAX / TensorFlow, PyTorch) that
        you wish to use for internal computations.

    :param node_to_edge_incidence:
        The node-to-edge incidence matrix of the graph. A 2D array of shape
        [n_edges, n_nodes], where n_edges is the number of edges in the graph.
        SciPy sparse matrices are also supported.
    
    :param edge_to_triangle_incidence:
        The edge-to-triangle incidence matrix of the graph. A 2D array of shape
        [n_triangles, n_edges], where n_triangles is the number of triangles in
        the graph. SciPy sparse matrices are also supported.

        Optional. If not provided or set to `None`, TODO.

    .. admonition:: Citation

        If you use this GeometricKernels space in your research, please consider
        citing :cite:t:`yang2024`.
    """

    def __init__(self,
                node_to_edge_incidence: B.Int,
                edge_to_triangle_incidence: Optional[B.Int] = None
        ):
        self.cache: Dict[int, Tuple[B.Numeric, B.Numeric]] = {}
        self._checks(node_to_edge_incidence, edge_to_triangle_incidence)
        self._set_laplacian(node_to_edge_incidence, edge_to_triangle_incidence)
        self.num_edges, self.num_vertices = node_to_edge_incidence.shape
        self.num_triangles = 0 if edge_to_triangle_incidence is None else edge_to_triangle_incidence.shape[0]

    @classmethod
    def from_adjacency(cls, adjacency_matrix) -> "GraphEdge":
        node_to_edge_incidence = TODO
        edge_to_triangle_incidence = TODO  # all possible triangles

        return cls(node_to_edge_incidence, edge_to_triangle_incidence)
        
    def __str__(self):
        return f"GraphEdge({self.num_vertices})"

    @staticmethod
    def _checks(node_to_edge_incidence, edge_to_triangle_incidence):
        """
        Checks if `node_to_edge_incidence` and `edge_to_triangle_incidence` are
        compatible and are of the appropriate format.
        """
        pass

    @property
    def dimension(self) -> int:
        """
        :return:
            0.
        """
        return 0  # this is needed for the kernel math to work out

    def _set_laplacian(self, node_to_edge_incidence, edge_to_triangle_incidence):
        """
        Construct the appropriate graph Laplacians.
        """
        self._down_laplacian = TODO
        self._up_laplacian = TODO
        self._laplacian = self._down_laplacian + self._up_laplacian

    def get_eigensystem(self, num):
        """
        Returns the first `num` eigenvalues and eigenvectors of the Hodge,
        Laplacian along with the indices of harmonic, curl, and gradient
        eigenvectors. Caches the solution to prevent re-computing the same values.

        :param num:
            Number of eigenpairs to return. Performs the computation at the
            first call. Afterwards, fetches the result from cache.
    
        :return:
            A dictionary with the following keys:
            - "evals": all eigenvalues, ordered by magnitude, from lowest to highest
            - "evecs": all respective eigenvectors
            - "harmonic_inds": list[int], indices of harmonic eigenvectors
            - "curl_inds": list[int], indices of curl eigenvectors
            - "gradient_inds": list[int], indices of gradient eigenvectors
        """
        if num not in self.cache:

            TODO

            self.cache[num] = {
                "evals": evals, # all eigenvalues, ordered by magnitude, from lowest to highest
                "evecs": evecs, # all respective eigenvectors
                "harmonic_inds": harmonic_inds, # list[int], indices of harmonic eigenvectors
                "curl_inds": curl_inds, # list[int], indices of curl eigenvectors
                "gradient_inds": gradient_inds, # list[int], indices of gradient eigenvectors
            }

        return self.cache[num]

    def get_eigenfunctions(self, num: int, type: Optional[str]) -> Eigenfunctions:
        """
        Returns the :class:`~.EigenfunctionsFromEigenvectors` object with `num`
        levels (i.e., in this case, `num` eigenpairs). If `type` is specified,
        returns only the eigenfunctions of that type.

        .. warning::
            If `type` is specified, the returned :class:`~.EigenfunctionsFromEigenvectors`
            object does not have to have `num` levels. It will typically have fewer. 

        :param num:
            Number of levels.
        :param type:
            Type of the eigenfunctions to return. Can be one of "harmonic",
            "curl" or "gradient".
        """
        eigensystem = self.get_eigensystem(num)

        inds = eigensystem[f"{type}_inds"] if type is not None else list(range(num))

        eigenfunctions = EigenfunctionsFromEigenvectors(
            TODO, # only choose the eigenvectors corresponding to the indices in `inds`
        )

        return eigenfunctions

    def get_eigenvalues(self, num: int, type: Optional[str]) -> B.Numeric:
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

        inds = eigensystem[f"{type}_inds"] if type is not None else list(range(num))

        eigenvalues = TODO  # only choose the eigenvalues corresponding to the indices in `inds`

        return eigenvalues

    def get_repeated_eigenvalues(self, num: int, type: Optional[str]) -> B.Numeric:
        """
        Same as :meth:`get_eigenvalues`.

        :param num:
            Same as :meth:`get_eigenvalues`.
        """
        return self.get_eigenvalues(num, type)

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

    @property
    def element_dtype(self):
        """
        :return:
            B.Int.
        """
        return B.Int
