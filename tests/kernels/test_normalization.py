from pathlib import Path

import numpy as np
import pytest

from geometric_kernels.kernels import MaternKarhunenLoeveKernel
from geometric_kernels.spaces.circle import Circle
from geometric_kernels.spaces.graph import Graph
from geometric_kernels.spaces.hypersphere import Hypersphere
from geometric_kernels.spaces.mesh import Mesh


@pytest.mark.parametrize("space_name", ["circle", "hypersphere", "mesh", "graph"])
def test_normalization_matern_kl_kernel(space_name):
    key = np.random.RandomState(1234)
    num_points = 300
    num_eigenfns = 10

    if space_name == "circle":
        # return
        space = Circle()
        key, points = space.random(key, num_points)
    elif space_name == "hypersphere":
        # return
        space = Hypersphere(2)
        key, points = space.random(key, num_points)
    elif space_name == "mesh":
        filename = Path(__file__).parent / "../teddy.obj"
        space = Mesh.load_mesh(str(filename))
        points = np.arange(space.num_vertices).reshape(-1, 1)
    elif space_name == "graph":
        A = np.array(
            [
                [0, 1, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        ).astype("float")

        space = Graph(A, normalize_laplacian=True)
        points = np.arange(space.num_vertices).reshape(-1, 1)

    else:
        return

    kernel = MaternKarhunenLoeveKernel(space, num_eigenfns, normalize=True)
    params, state = kernel.init_params_and_state()
    params = {"nu": np.r_[2.5], "lengthscale": np.r_[1.0]}

    kxx = kernel.K_diag(params, state, points)
    np.testing.assert_allclose(np.mean(kxx), 1.0)

    kxx = np.diag(kernel.K(params, state, points))
    np.testing.assert_allclose(np.mean(kxx), 1.0)
