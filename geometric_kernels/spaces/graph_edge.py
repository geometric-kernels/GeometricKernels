"""
This module provides the :class:`EdgeGraph` space.
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
        
    :param edge_triangle_incidence_matrix: optional
        An m x t dimensional matrix where m is the number of edges and t is the number of triangles.
        edge_triangle_incidence_matrix[e, t] is -1 if edge e is anti-aligned with triangle t, 1 if edge e is aligned with triangle t, 0 otherwise.
        
    :param edge_laplacian: 
        An m x m dimensional matrix where m is the number of edges.
        edge_laplcian is computed as 
            incidence_matrix.T @ incidence_matrix, if edge_triangle_incidence_matrix is None
            incidence_matrix.T @ incidence_matrix + edge_triangle_incidence_matrix @ edge_triangle_incidence_matrix.T, if edge_triangle_incidence_matrix is not None

    .. admonition:: Citation

        If you use this GeometricKernels space in your research, please consider
        citing :cite:t:`yang2024`.
    """

    def __init__(self, incidence_matrix: B.Numeric, edge_triangle_incidence_matrix=None):  # type: ignore
        self.cache: Dict[int, Tuple[B.Numeric, B.Numeric]] = {}
        self.incidence_matrix = incidence_matrix
        if edge_triangle_incidence_matrix is not None:
            self.edge_triangle_incidence_matrix = edge_triangle_incidence_matrix
        else:
            self.edge_triangle_incidence_matrix = None

        self._checks(incidence_matrix, self.edge_triangle_incidence_matrix)  
        self._set_laplacian(incidence_matrix)  # type: ignore
 

    @staticmethod
    def _checks(incidence, edge_triangle_incidence_matrix=None):
        """
        Checks if 'incidence' has dimension n x m and if each column has two non-zero entries.
        """
        nnz = np.count_nonzero(incidence, axis=0)
        assert np.all(nnz == 2), "Each column of the incidence matrix should have exactly two non-zero entries."
        if edge_triangle_incidence_matrix is not None:
            nnz2 = np.count_nonzero(edge_triangle_incidence_matrix, axis=0)
            assert np.all(nnz2 == 3), "Each column of the edge triangle incidence matrix should have exactly 3 non-zero entries."


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
        return self.incidence_matrix.shape[0]
    
    @property 
    def num_edges(self) -> int:
        """
        Number of edges in the graph.
        """
        return self.incidence_matrix.shape[1]
    
    @property
    def num_triangles(self) -> int:
        """
        Number of triangles in the graph.
        """
        if self.edge_triangle_incidence_matrix is None:
            return 0
        else: 
            return self.edge_triangle_incidence_matrix.shape[1]

    def _set_laplacian(self, incidence_matrix):
        """
        Construct the appropriate graph Laplacian from the adjacency matrix.
        """
        if self.edge_triangle_incidence_matrix is None:
            self._edge_laplacian = incidence_matrix.T @ incidence_matrix
        else: 
            self._down_edge_laplacian = incidence_matrix.T @ incidence_matrix
            self._up_edge_laplacian = self.edge_triangle_incidence_matrix @ self.edge_triangle_incidence_matrix.T
            self._edge_laplacian = self._down_edge_laplacian + self._up_edge_laplacian
            
        
    @property
    def edge_laplacian(self):
        """
        Return the Edge Laplacian
        """
        if self.edge_triangle_incidence_matrix is None:
            return self._edge_laplacian
        else:
            return self._edge_laplacian, self._down_edge_laplacian, self._up_edge_laplacian
       

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
        eps = np.finfo(float).eps
        if num not in self.cache:
            evals, evecs = eigenpairs(self._edge_laplacian, num)
            if self.edge_triangle_incidence_matrix is not None:
                # harmonic ones are the ones associated to zero eigenvalues of the edge laplacian
                total_var = []
                total_div = []
                total_curl = []
                num_eigemodes = len(evals)
                for i in range(num_eigemodes):
                    total_var.append(evecs[:, i].T@self._edge_laplacian@evecs[:, i])
                    total_div.append(evecs[:, i].T@self._down_edge_laplacian@evecs[:, i])
                    total_curl.append(evecs[:, i].T@self._up_edge_laplacian@evecs[:, i])
                    
                harm_evecs = np.where(np.array(total_var) < eps)[0]
                grad_evecs = np.where(np.array(total_div) > eps)[0]
                curl_eflow = np.where(np.array(total_curl) > eps)[0]
                assert len(harm_evecs) + len(grad_evecs) + len(curl_eflow) == num_eigemodes, "The eigenmodes are not correctly organized"
                evals = np.concatenate((evals[harm_evecs], evals[grad_evecs], evals[curl_eflow]))
                evecs = np.concatenate((evecs[:, harm_evecs], evecs[:, grad_evecs], evecs[:, curl_eflow]), axis=1)

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
        num_edges = B.shape(self._edge_laplacian)[0]
        key, random_edges = B.randint(
            key, dtype_integer(key), number, 1, lower=0, upper=num_edges
        )

        return key, random_edges

    @property
    def element_shape(self):
        """
        :return:
            [1].
        """
        return [1]
