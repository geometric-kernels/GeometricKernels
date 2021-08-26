from pathlib import Path

import numpy as np
from pytest import fixture

from geometric_kernels.kernels import MeshKernel
from geometric_kernels.spaces import Mesh

_TRUNCATION_LEVEL = 10
_NU = 1 / 2.0


@fixture(name="mesh_kernel")
def fixture_mesh_kernel() -> MeshKernel:
    filename = Path(__file__).parent / "../teddy.obj"
    mesh = Mesh.load_mesh(str(filename))
    return MeshKernel(mesh, _NU, _TRUNCATION_LEVEL)


def test_eigenvalues(mesh_kernel: MeshKernel):
    assert mesh_kernel.eigenvalues(lengthscale=0.81).shape == (_TRUNCATION_LEVEL, 1)


def test_eigenfunctions(mesh_kernel: MeshKernel):
    num_data = 11
    Phi = mesh_kernel.eigenfunctions(lengthscale=0.93)
    X = np.random.randint(low=0, high=mesh_kernel.space.num_vertices, size=(num_data, 1))

    assert Phi(X).shape == (num_data, _TRUNCATION_LEVEL)


def test_K_shapes(mesh_kernel: MeshKernel):
    N1, N2 = 11, 13
    X = np.random.randint(low=0, high=mesh_kernel.space.num_vertices, size=(N1, 1))
    X2 = np.random.randint(low=0, high=mesh_kernel.space.num_vertices, size=(N2, 1))

    K = mesh_kernel.K(X, None, lengthscale=0.99)
    assert K.shape == (N1, N1)

    K = mesh_kernel.K(X, X2, lengthscale=0.99)
    assert K.shape == (N1, N2)

    K = mesh_kernel.K_diag(X, lengthscale=0.99)
    assert K.shape == (N1,)
