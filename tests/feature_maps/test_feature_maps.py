from pathlib import Path

import numpy as np
import pytest

from geometric_kernels.kernels import MaternGeometricKernel, default_feature_map
from geometric_kernels.spaces.circle import Circle
from geometric_kernels.spaces.graph import Graph
from geometric_kernels.spaces.hyperbolic import Hyperbolic
from geometric_kernels.spaces.hypersphere import Hypersphere
from geometric_kernels.spaces.mesh import Mesh
from geometric_kernels.spaces.spd import SymmetricPositiveDefiniteMatrices


@pytest.mark.parametrize(
    "space_name", ["circle", "hypersphere", "mesh", "graph", "hyperbolic", "spd"]
)
def test_feature_maps(space_name):
    key = np.random.RandomState(1234)
    num_points = 5

    if space_name == "circle":
        space = Circle()
        kwargs = {}
    elif space_name == "hypersphere":
        space = Hypersphere(2)
        kwargs = {}
    elif space_name == "mesh":
        filename = Path(__file__).parent / "../teddy.obj"
        space = Mesh.load_mesh(str(filename))
        kwargs = {}
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
        kwargs = {}
    elif space_name == "hyperbolic":
        space = Hyperbolic(dim=2)
        kwargs = {"key": key}
    elif space_name == "spd":
        space = SymmetricPositiveDefiniteMatrices(n=2)
        kwargs = {"key": key}
    else:
        raise ValueError(f"Unknown space {space}")

    kernel = MaternGeometricKernel(space, **kwargs)
    params = kernel.init_params()
    params = {"nu": np.r_[2.5], "lengthscale": np.r_[1.0]}

    feature_map = default_feature_map(kernel=kernel)

    key, points = space.random(key, num_points)

    _, embedding = feature_map(points, params, **kwargs)

    kernel_mat = kernel.K(params, points, points)
    kernel_mat_alt = np.matmul(embedding, embedding.T)

    np.testing.assert_allclose(kernel_mat, kernel_mat_alt)
