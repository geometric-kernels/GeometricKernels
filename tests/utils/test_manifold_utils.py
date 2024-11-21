import numpy as np
import pytest

from geometric_kernels.spaces import Hyperbolic
from geometric_kernels.utils.manifold_utils import hyperbolic_distance

from ..helper import check_function_with_backend


@pytest.mark.parametrize("dim", [2, 3, 9, 10])
@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_hyperboloid_distance(dim, backend):
    space = Hyperbolic(dim=dim)

    key = np.random.RandomState(0)
    N, N2 = key.randint(low=2, high=15, size=2)
    key, X = space.random(key, N)
    key, X2 = space.random(key, N2)

    X_expanded = np.tile(X[..., None, :], (1, X2.shape[0], 1))  # (N, M, n+1)
    X2_expanded = np.tile(X2[None], (X.shape[0], 1, 1))  # (N, M, n+1)
    result = space.metric.dist(X_expanded, X2_expanded)

    # Check that our implementation of the hyperbolic distance coincides with
    # the one from geomstats.
    check_function_with_backend(
        backend,
        result,
        hyperbolic_distance,
        X,
        X2,
    )
