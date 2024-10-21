from pathlib import Path

import numpy as np
import pytest

from geometric_kernels.spaces import Mesh


@pytest.fixture()
def mesh() -> Mesh:
    filename = Path(__file__).parent / "../teddy.obj"
    mesh = Mesh.load_mesh(str(filename))
    return mesh


def test_mesh_shapes():
    Nv = 11  # num vertices
    Nf = 13  # num faces
    dim = 3  # ambient dimension
    vertices = np.random.randn(Nv, dim)
    faces = np.random.randint(0, Nv, size=(Nf, 3))
    mesh = Mesh(vertices=vertices, faces=faces)
    assert mesh.vertices.shape == (Nv, dim)
    assert mesh.faces.shape == (Nf, 3)


def test_read_mesh(mesh: Mesh):
    assert mesh.vertices.shape == (mesh.num_vertices, mesh.dimension + 1)
    assert mesh.faces.shape == (mesh.num_faces, 3)


def test_eigenvalues(mesh: Mesh):
    assert mesh.get_eigenvalues(10).shape == (10, 1)
    assert mesh.get_eigenvalues(13).shape == (13, 1)


def test_eigenvectors(mesh: Mesh):
    assert mesh.get_eigenvectors(10).shape == (mesh.num_vertices, 10)
    assert mesh.get_eigenvectors(13).shape == (mesh.num_vertices, 13)
    assert set(mesh.cache.keys()) == set([10, 13])
