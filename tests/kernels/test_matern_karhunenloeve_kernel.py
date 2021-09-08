from pathlib import Path

import numpy as np
from pytest import fixture

from geometric_kernels.kernels import MaternKarhunenLoeveKernel
from geometric_kernels.spaces import Mesh

_TRUNCATION_LEVEL = 10
_NU = 1 / 2.0


@fixture(name="kernel")
def fixture_mesh_kernel() -> MaternKarhunenLoeveKernel:
    filename = Path(__file__).parent / "../teddy.obj"
    mesh = Mesh.load_mesh(str(filename))
    return MaternKarhunenLoeveKernel(mesh, _NU, _TRUNCATION_LEVEL)


def test_eigenvalues(kernel: MaternKarhunenLoeveKernel):
    assert kernel.eigenvalues(lengthscale=np.r_[0.81]).shape == (_TRUNCATION_LEVEL, 1)


def test_eigenfunctions(kernel: MaternKarhunenLoeveKernel):
    num_data = 11
    Phi = kernel.eigenfunctions()
    X = np.random.randint(low=0, high=kernel.space.num_vertices, size=(num_data, 1))

    assert Phi(X, lengthscale=np.r_[0.93]).shape == (num_data, _TRUNCATION_LEVEL)


def test_K_shapes(kernel: MaternKarhunenLoeveKernel):
    N1, N2 = 11, 13
    X = np.random.randint(low=0, high=kernel.space.num_vertices, size=(N1, 1))
    X2 = np.random.randint(low=0, high=kernel.space.num_vertices, size=(N2, 1))

    K = kernel.K(X, None, lengthscale=np.r_[0.99])
    assert K.shape == (N1, N1)

    K = kernel.K(X, X2, lengthscale=np.r_[0.99])
    assert K.shape == (N1, N2)

    K = kernel.K_diag(X, lengthscale=np.r_[0.99])
    assert K.shape == (N1,)
