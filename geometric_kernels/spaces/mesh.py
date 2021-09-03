"""
Mesh object
"""
from typing import Any, Callable, Dict, Tuple

import eagerpy as ep
import numpy as np
import potpourri3d as pp3d
import robust_laplacian
import scipy.sparse.linalg as sla

from geometric_kernels.spaces import SpaceWithEigenDecomposition
from geometric_kernels.types import TensorLike
from geometric_kernels.utils import cast_to_int, take_along_axis


class ConvertEigenvectorsToEigenfunctions:
    """
    Converts the array of eigenvectors to a callable objects,
    where inputs are given by the indices.
    """

    def __init__(self, eigenvectors: np.ndarray):
        # Always numpy to seamleassy convert to a desired backend
        assert isinstance(eigenvectors, np.ndarray)
        self.eigenvectors_np = eigenvectors
        self.eigenvectors = None

    def __call__(self, indices: TensorLike) -> TensorLike:
        """
        Selects N locations from the  eigenvectors.

        :param indices: indices [N, 1]
        :return: [N, L]
        """
        # Convert stored numpy eigenvectors to whatever indices have as a backend
        indices = ep.astensor(indices)

        if not isinstance(indices, type(self.eigenvectors)):
            self.eigenvectors = ep.from_numpy(indices, self.eigenvectors_np)

        assert len(indices.shape) == 2
        assert indices.shape[-1] == 1
        indices = cast_to_int(indices)

        # This is a very hacky way of taking along 0'th axis.
        # For some reason eagerpy does not take along axis other than last.
        Phi = take_along_axis(self.eigenvectors, indices, axis=0)
        return Phi


class Mesh(SpaceWithEigenDecomposition):
    """
    A representation of a surface mesh. Mimics `PyMesh` interface. Uses
    `potpourri3d` to read mesh files.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        """
        :param vertices: A [Nv, D] array of vertex coordinates, where Nv is the number of vertices,
            D is the dimention of the embedding space (D must be either 2 or 3).
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
            evals, evecs = sla.eigsh(L, num, M, sigma=1e-8)
            evecs, _ = np.linalg.qr(evecs)
            self.cache[num] = (evecs, evals.reshape(-1, 1))

        return self.cache[num]

    def get_eigenvectors(self, num: int) -> TensorLike:
        """
        :param num: number of eigenvectors returned
        :return: eigenvectors [Nv, num]
        """
        return self.get_eigensystem(num)[0]

    def get_eigenvalues(self, num: int) -> TensorLike:
        """
        :param num: number of eigenvalues returned
        :return: eigenvalues [num, 1]
        """
        return self.get_eigensystem(num)[1]

    def get_eigenfunctions(self, num: int) -> Callable[[TensorLike], TensorLike]:
        """
        First `num` eigenfunctions of the Laplace-Beltrami operator on the Mesh.

        :param num: number of eigenfunctions returned
        :return: eigenfu [Nv, num]
        """
        eigenfunctions = ConvertEigenvectorsToEigenfunctions(self.get_eigenvectors(num))
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
        """Dimension, D"""
        return self._vertices.shape[1]

    @property
    def vertices(self) -> np.ndarray:
        """
        A [Nv, D] array of vertex coordinates, where Nv is the number of vertices,
        D is the dimention of the embedding space (D must be either 2 or 3).
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
