"""
This module provides the :class:`Mesh` space.
"""

import lab as B
import numpy as np
import potpourri3d as pp3d
import robust_laplacian
import scipy.sparse.linalg as sla
from beartype.typing import Dict, Tuple
from scipy.linalg import eigh

from geometric_kernels.lab_extras import dtype_integer
from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import (
    Eigenfunctions,
    EigenfunctionsFromEigenvectors,
)


class Mesh(DiscreteSpectrumSpace):
    """
    The GeometricKernels space representing the node set of any user-provided mesh.

    We only support the commonly used 2-dimensional meshes (discrete counterparts
    of surfaces, 2-dimensional manifolds in a 3-dimensional ambient space) and
    1-dimensional meshes (discrete counterparts of curves, 1-dimensional manifolds).

    We use `potpourri3d <https://github.com/nmwsharp/potpourri3d>`_ to load meshes
    and mimic the interface of `PyMesh <https://github.com/PyMesh/PyMesh>`_.

    The elements of this space are represented by node indices, integer values
    from 0 to n-1, where n is the number of nodes in the user-provided mesh.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        """
        :param vertices: A [Nv, D] array of vertex coordinates, where Nv is the number of vertices,
            D is the dimension of the embedding space (D must be either 2 or 3).
            Note that this corresponds to a (D-1)-dimensional mesh, a discretization of some
            assumed (D-1)-dimensional manifold.
        :param faces: A [Nf, 3] array of vertex indices that represents a
            generalized array of faces, where Nf is the number of faces.

            .. Note:
                Only 3 vertex indices per face are supported
        """
        self._vertices = vertices
        self._faces = faces
        self._eigenvalues = None
        self._eigenfunctions = None
        self.cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def get_eigensystem(self, num: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the first `num` eigenvalues and eigenfunctions of the Laplace-Beltrami
        operator on the space. Makes use of Nick Sharp's robust laplacian package
        and Scipy's sparse linear algebra.

        Caches the solution to prevent re-computing the same values.

        TODO(VD): Make sure this is the optimal way to compute this!

        :param num: number of eigenvalues and functions to return.
        :return: A Tuple of eigenvectors [Nv, num], eigenvalues [num, 1]
        """
        if num not in self.cache:
            L, M = robust_laplacian.mesh_laplacian(self.vertices, self.faces)
            if L.shape[0] == num:
                evals, evecs = eigh(L.toarray(), M.toarray())
            else:
                evals, evecs = sla.eigsh(L, num, M, sigma=1e-8)
            evecs, _ = np.linalg.qr(evecs)
            evecs *= np.sqrt(self.num_vertices)
            evals = np.clip(
                evals, a_min=0.0, a_max=None
            )  # prevent small negative values
            self.cache[num] = (evecs, evals.reshape(-1, 1))

        return self.cache[num]

    def get_eigenvectors(self, num: int) -> B.Numeric:
        """
        :param num: number of eigenvectors returned
        :return: eigenvectors [Nv, num]
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
        :param num: number of eigenvalues returned
        :return: eigenvalues [num, 1]
        """
        return self.get_eigenvalues(num)

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        First `num` eigenfunctions of the Laplace-Beltrami operator on the Mesh.

        :param num: number of eigenfunctions returned
        :return: eigenfunctions [Nv, num]
        """
        eigenfunctions = EigenfunctionsFromEigenvectors(self.get_eigenvectors(num))
        return eigenfunctions

    @property
    def num_vertices(self) -> int:
        """Number of vertices, Nv"""
        return len(self._vertices)

    @property
    def num_faces(self) -> int:
        """Number of faces, Nf"""
        return len(self._faces)

    @property
    def dimension(self) -> int:
        """
        Dimension of the space. Equal to D-1, where D is the dimension of the embedding space.
        """
        return self._vertices.shape[1] - 1

    @property
    def vertices(self) -> np.ndarray:
        """
        A [Nv, D] array of vertex coordinates, where Nv is the number of vertices,
        D is the dimension of the embedding space (D must be either 2 or 3).
        """
        return self._vertices

    @property
    def faces(self) -> np.ndarray:
        """
        A [Nf, 3] array of vertex indices that represents a generalized array of
        faces, where Nf is the number of faces.
        """
        return self._faces

    @classmethod
    def load_mesh(cls, filename: str) -> "Mesh":
        """
        :param filename: path to read the file from. Supported formats: `obj`,
            `ply`, `off`, and `stl`. Format inferred automatically from the path
            extension.
        """
        # load vertices and faces using potpourri3d
        vertices, faces = pp3d.read_mesh(filename)
        # return Mesh
        return cls(vertices, faces)

    def random(self, key, number):
        key, random_vertices = B.randint(
            key, dtype_integer(key), number, 1, lower=0, upper=self.num_vertices
        )
        return key, random_vertices
