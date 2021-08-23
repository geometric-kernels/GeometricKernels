from pathlib import Path

import numpy as np
from pytest import fixture

from geometric_kernels.spaces import Mesh


@fixture(name="mesh")
def fixture_get_dummy_mesh() -> Mesh:
    Nv = 11  # num vertices
    Nf = 13  # num faces
    dim = 3  # dimension
    vertices = np.random.randn(Nv, dim)
    faces = np.random.randint(0, Nv, size=(Nf, 3))
    return Mesh(vertices=vertices, faces=faces)


def test_mesh_shapes(mesh):
    assert mesh.vertices.shape == (mesh.num_vertices, mesh.dim)
    assert mesh.faces.shape == (mesh.num_faces, 3)


def test_read_mesh():
    filename = Path(__file__).parent / "teddy.obj"
    mesh = Mesh.load_mesh(str(filename))
    assert mesh.vertices.shape == (mesh.num_vertices, mesh.dim)
    assert mesh.faces.shape == (mesh.num_faces, 3)
