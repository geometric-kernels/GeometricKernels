from pathlib import Path

import numpy as np
import pytest

from geometric_kernels.feature_maps import RandomPhaseFeatureMapNoncompact
from geometric_kernels.kernels import MaternFeatureMapKernel, MaternKarhunenLoeveKernel
from geometric_kernels.spaces.circle import Circle
from geometric_kernels.spaces.graph import Graph
from geometric_kernels.spaces.hyperbolic import Hyperbolic
from geometric_kernels.spaces.hypersphere import Hypersphere
from geometric_kernels.spaces.mesh import Mesh
from geometric_kernels.spaces.spd import SymmetricPositiveDefiniteMatrices


@pytest.mark.parametrize("space_name", ["circle", "hypersphere", "mesh", "graph"])
def test_normalization_matern_kl_kernel(space_name):
    key = np.random.RandomState(1234)
    num_points = 300
    num_eigenfns = 10

    if space_name == "circle":
        space = Circle()
        key, points = space.random(key, num_points)
    elif space_name == "hypersphere":
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
        raise ValueError(f"Unknown space {space}")

    kernel = MaternKarhunenLoeveKernel(space, num_eigenfns, normalize=True)
    params = kernel.init_params()
    params = {"nu": np.r_[2.5], "lengthscale": np.r_[1.0]}

    kxx = kernel.K_diag(params, points)
    np.testing.assert_allclose(np.mean(kxx), 1.0)

    kxx = np.diag(kernel.K(params, points))
    np.testing.assert_allclose(np.mean(kxx), 1.0)


@pytest.mark.parametrize("space_name", ["hyperbolic", "spd"])
def test_normalization_feature_map_kernel(space_name):
    key = np.random.RandomState(1234)
    num_points = 300
    num_features = 10

    if space_name == "hyperbolic":
        space = Hyperbolic(dim=2)
        points = space.random_point(num_points)
    elif space_name == "spd":
        space = SymmetricPositiveDefiniteMatrices(n=2)
        points = space.random_point(num_points)
    else:
        raise ValueError(f"Unknown space {space}")

    params = dict(nu=np.r_[2.5], lengthscale=np.r_[1.0])

    feature_map = RandomPhaseFeatureMapNoncompact(space, num_features)

    kernel = MaternFeatureMapKernel(space, feature_map, key)

    kxx = kernel.K_diag(params, points)
    np.testing.assert_allclose(kxx, 1.0)

    kxx = np.diag(kernel.K(params, points))
    np.testing.assert_allclose(kxx, 1.0)
