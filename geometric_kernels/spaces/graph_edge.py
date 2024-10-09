"""
This module provides the :class:`EdgeGraph` space.
"""

import lab as B
import numpy as np
import networkx as nx
from beartype.typing import Dict, Tuple, Optional

from geometric_kernels.lab_extras import (
    degree,
    dtype_integer,
    eigenpairs,
    reciprocal_no_nan,
    set_value,
    take_along_axis,
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

    def __init__(self, G, triangle_list=None, sc_lifting=False):  # type: ignore
        self.G = G 
        self.cache: Dict[int, Tuple[B.Numeric, B.Numeric]] = {}
        self.incidence_matrix = nx.incidence_matrix(G, oriented=True).toarray() # "obtain the oriented incidence matrix"
        if sc_lifting is not False:
            if triangle_list is None:
                print("No list of triangles is provided, we consider all triangles in the graph as 2-simplices.")
                self.triangles = self.triangles_all_clique()
                self.triangle_incidence_matrix = self.triangles_to_B2()
            else:
                self.triangles = triangle_list
                self.triangle_incidence_matrix = self.triangles_to_B2()
        else:
            self.triangle_incidence_matrix = None

        self._checks(self.incidence_matrix, self.triangle_incidence_matrix) 
        self._set_laplacian()  # type: ignore
        self.num_vertices, self.num_edges = self.incidence_matrix.shape
        self.num_triangles = self.triangle_incidence_matrix.shape[1] if self.triangle_incidence_matrix is not None else 0
 

    @staticmethod
    def _checks(incidence_matrix, triangle_incidence_matrix=None):
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
        
    def sc_simplices(self):
        """return the nodes, edges and triangles in the graph"""
        print('----Simplicial 2-complex summary---')
        print('nodes: ', list(self.G.nodes))
        print('edges: ', list(self.G.edges))
        print('triangles: ', self.triangles)
        return None 

    def _set_laplacian(self):
        """
        Construct the appropriate graph Laplacian from the adjacency matrix.
        """
        self._down_edge_laplacian = self.incidence_matrix.T @ self.incidence_matrix
        if self.triangle_incidence_matrix is None:
            self._up_edge_laplacian = B.zeros(
                B.dtype(self._down_edge_laplacian), *B.shape(self._down_edge_laplacian)
            )
        else:
            self._up_edge_laplacian = B.matmul(
                self.triangle_incidence_matrix, self.triangle_incidence_matrix, tr_b=True
            )
        self._edge_laplacian = self._down_edge_laplacian + self._up_edge_laplacian
            
    @property
    def incidence_matrices(self):
        """
        Return the incidence matrix
        """
        if self.triangle_incidence_matrix is None:
            return self.incidence_matrix
        else:
            return self.incidence_matrix, self.triangle_incidence_matrix

    @property
    def edge_laplacian(self):
        """
        Return the Edge Laplacian
        """
        if self.triangle_incidence_matrix is None:
            return self._edge_laplacian
        else:
            return self._edge_laplacian, self._down_edge_laplacian, self._up_edge_laplacian
       
    def triangles_all_clique(self) -> list:
        """
        Get a list of triangles in the graph.

        Returns:
            list: List of triangles.
        """
        cliques = nx.enumerate_all_cliques(self.G)
        triangle_vertices = [x for x in cliques if len(x) == 3]
        # sort the triangles
        triangle_vertices = [sorted(tri) for tri in triangle_vertices]
        return triangle_vertices
    
    def triangles_to_B2(self) -> np.ndarray:
        """
        Create the B2 matrix (edge-triangle) from the triangles.
        
        The `triangle_incidence_matrix` parameter is a `numpy` array of shape `(n_edges, n_triangles)` where `n_triangles` is the number of triangles in the graph.

        The entry `triangle_incidence_matrix[e, t]` is
        - `-1` if edge `e` is anti-aligned with triangle `t`, e.g., $(1,3)$ is anti-aligned with $(1,2,3)$,
        - `1` if edge `e` is aligned with triangle `t`, e.g., $(1,2)$ is aligned with $(1,2,3)$, and
        - `0` otherwise.

        Args:
            triangles (list): List of triangles.
            edges (list): List of edges.

        Returns:
            np.ndarray: B2 matrix.
        """
        edges = list(self.G.edges)
        triangles = self.triangles
        B2 = np.zeros((len(edges), len(triangles)))
        for j, triangle in enumerate(triangles):
            a, b, c = triangle
            try:
                index_a = edges.index((a, b))
            except ValueError:
                index_a = edges.index((b, a))
            try:
                index_b = edges.index((b, c))
            except ValueError:
                index_b = edges.index((c, b))
            try:
                index_c = edges.index((a, c))
            except ValueError:
                index_c = edges.index((c, a))

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
            evals, evecs = eigenpairs(self._edge_laplacian, num)
            # make a dictionary of the eigenvalues and eigenvectors where the keys are the indices of the eigenvalues, and add another value which indicates the type of the eigenbasis (harmonic, gradient, curl)
            self.hodge_eigenbasis = {}
            # add keys to the dictionary
            if self.triangle_incidence_matrix is not None:
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
                    
                for i in range(num_eigemodes):
                    if total_var[i] < eps:
                        # harm_evecs.append(i)
                        hodge_type = 'harmonic'
                    elif total_div[i] > eps:
                        # grad_evecs.append(i)
                        hodge_type = 'gradient'
                    elif total_curl[i] > eps:
                        # curl_evecs.append(i)
                        hodge_type = 'curl'
                    self.hodge_eigenbasis[i] = {'eval': evals[i], 'evec': evecs[:, i], 'type': hodge_type}
                        
            self.cache[num] = self.hodge_eigenbasis

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
        return np.array([entry['evec'] for entry in self.get_eigensystem(num).values()]).T

    def get_eigenvalues(self, num: int) -> B.Numeric:
        """
        :param num:
            Number of eigenvalues to return.

        :return:
            Array of eigenvalues, with shape [num, 1].
        """
        return np.array([entry['eval'] for entry in self.get_eigensystem(num).values()])[:,None]
    
    def get_eigenbasis_type(self, num: int): 
        """
        :param num:
            Number of eigenvalues to return.

        :return:
            Array of strings, with shape [num, 1].
        """
        return [entry['type'] for entry in self.get_eigensystem(num).values()]
    
    # get particular type of eigenbasis
    def get_eigenfunctions_by_type(self, num: int, hodge_type: str) -> Eigenfunctions:
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
        eigenfunctions = EigenfunctionsFromEigenvectors(self.get_eigenvectors_by_type(num, hodge_type))
        return eigenfunctions
    
    def get_eigenvectors_by_type(self, num: int, hodge_type: str) -> B.Numeric:
        """
        :param num:
            Number of eigenvectors to return.
        :param hodge_type:
            The type of the eigenbasis. It can be 'harmonic', 'gradient', or 'curl'.

        :return:
            Array of eigenvectors, with shape [n, num].
        """
        assert hodge_type in ['harmonic', 'gradient', 'curl'] #"The hodge type should be either 'harmonic', 'gradient', or 'curl'."
        eigenbasis = self.get_eigensystem(num)
        return np.array([entry['evec'] for entry in eigenbasis.values() if entry['type'] == hodge_type]).T
    
    def get_eigenvalues_by_type(self, num: int, hodge_type: str) -> B.Numeric:
        """
        :param num:
            Number of eigenvalues to return.
        :param hodge_type:
            The type of the eigenbasis. It can be 'harmonic', 'gradient', or 'curl'.

        :return:
            Array of eigenvalues, with shape [num, 1].
        """
        assert hodge_type in ['harmonic', 'gradient', 'curl'] # "The hodge type should be either 'harmonic', 'gradient', or 'curl'."
        eigenbasis = self.get_eigensystem(num)
        return np.array([entry['eval'] for entry in eigenbasis.values() if entry['type'] == hodge_type])[:,None]
    

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
            key, dtype_integer(key), number, 1, lower=0, upper=num_edges,
        )

        return key, random_edges

    @property
    def element_shape(self):
        """
        :return:
            [1].
        """
        return [1]
