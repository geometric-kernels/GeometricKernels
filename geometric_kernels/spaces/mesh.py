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
    The GeometricKernels space representing the node set of any
    user-provided mesh.

    The elements of this space are represented by node indices, integer values
    from 0 to Nv-1, where Nv is the number of nodes in the user-provided mesh.

    Each individual eigenfunction constitutes a *level*.

    .. note::
        We only support the commonly used 2-dimensional meshes (discrete
        counterparts of surfaces, 2-dimensional manifolds in a 3-dimensional
        ambient space).

    .. note::
        A tutorial on how to use this space is available in the
        :doc:`Mesh.ipynb </examples/Mesh>` notebook.

    .. note::
        We use `potpourri3d <https://github.com/nmwsharp/potpourri3d>`_ to
        load meshes and mimic the interface of
        `PyMesh <https://github.com/PyMesh/PyMesh>`_.

    :param vertices:
        A [Nv, 3] array of vertex coordinates, Nv is the number of vertices.
    :param faces:
        A [Nf, 3] array of vertex indices that represents a
        generalized array of faces, where Nf is the number of faces.

        .. Note:
            Only 3 vertex indices per face are supported, i.e. mesh must be
            triangulated.

    .. admonition:: Citation

        If you use this GeometricKernels space in your research, please consider
        citing :cite:t:`borovitskiy2020`.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self._vertices = vertices
        assert self._vertices.shape[1] == 3  # make sure we all is in R^3.
        self._faces = faces
        self._eigenvalues = None
        self._eigenfunctions = None
        self.cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def __str__(self):
        return f"Mesh({self.num_vertices})"

    def get_eigensystem(self, num: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the first `num` eigenvalues and eigenvectors of the `robust
        Laplacian <https://github.com/nmwsharp/nonmanifold-laplacian>`_.
        Caches the solution to prevent re-computing the same values.

        .. note::
            If the `adjacency_matrix` was a sparse SciPy array, requesting
            **all** eigenpairs will lead to a conversion of the sparse matrix
            to a dense one due to scipy.sparse.linalg.eigsh limitations.

        .. warning::
            Always uses SciPy (thus CPU) for internal computations. We will
            need to fix this in the future.

        .. todo::
            See warning above.

        :param num:
            Number of eigenpairs to return. Performs the computation at the
            first call. Afterwards, fetches the result from cache.

        :return:
            A tuple of eigenvectors [nv, num], eigenvalues [num, 1].
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
        :param num:
            Number of eigenvectors to return.

        :return:
            Array of eigenvectors, with shape [Nv, num].
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

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        Returns the :class:`~.EigenfunctionsFromEigenvectors` object with
        `num` levels (i.e., in this case, `num` eigenpairs).

        :param num:
            Number of levels.
        """
        eigenfunctions = EigenfunctionsFromEigenvectors(self.get_eigenvectors(num))
        return eigenfunctions

    @property
    def num_vertices(self) -> int:
        """
        Number of vertices in the mesh, Nv.
        """
        return len(self._vertices)

    @property
    def num_faces(self) -> int:
        """
        Number of faces in the mesh, Nf.
        """
        return len(self._faces)

    @property
    def dimension(self) -> int:
        """
        :return:
            2.
        """
        return 2

    @property
    def vertices(self) -> np.ndarray:
        """
        A [Nv, 3] array of vertex coordinates, Nv is the number of vertices.
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
        Construct :class:`Mesh` by loading a mesh from the file at `filename`.

        :param filename:
            Path to read the file from. Supported formats: `obj`,
            `ply`, `off`, and `stl`. Format inferred automatically from the
            file extension.

        :return:
            And object of class :class:`Mesh` representing the loaded mesh.
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
