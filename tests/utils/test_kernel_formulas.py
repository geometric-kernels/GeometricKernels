from math import log, tanh

import numpy as np
import pytest
from sklearn.metrics.pairwise import rbf_kernel

from geometric_kernels.spaces import HypercubeGraph
from geometric_kernels.utils.kernel_formulas import hypercube_graph_heat_kernel

from ..helper import check_function_with_backend


@pytest.mark.parametrize("d", [1, 5, 10])
@pytest.mark.parametrize("lengthscale", [1.0, 5.0, 10.0])
@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_hypercube_graph_heat_kernel(d, lengthscale, backend):
    space = HypercubeGraph(d)

    key = np.random.RandomState(0)
    N, N2 = key.randint(low=1, high=min(2**d, 10) + 1, size=2)
    key, X = space.random(key, N)
    key, X2 = space.random(key, N2)

    gamma = -log(tanh(lengthscale**2 / 2))
    result = rbf_kernel(X, X2, gamma=gamma)

    def heat_kernel(lengthscale, X, X2):
        return hypercube_graph_heat_kernel(
            lengthscale, X, X2, normalized_laplacian=False
        )

    # Checks that the heat kernel on the hypercube graph coincides with the RBF
    # restricted onto binary vectors, with appropriately redefined length scale.
    check_function_with_backend(
        backend,
        result,
        heat_kernel,
        np.array([lengthscale]),
        X,
        X2,
        atol=1e-2,
    )

    if d > 5:
        X_first = X[0:1, :3]
        X2_first = X2[0:1, :3]
        X_second = X[0:1, 3:]
        X2_second = X2[0:1, 3:]

        K_first = hypercube_graph_heat_kernel(
            np.array([lengthscale]), X_first, X2_first, normalized_laplacian=False
        )
        K_second = hypercube_graph_heat_kernel(
            np.array([lengthscale]), X_second, X2_second, normalized_laplacian=False
        )

        result = K_first * K_second

        # Checks that the heat kernel of the product is equal to the product
        # of heat kernels.
        check_function_with_backend(
            backend,
            result,
            heat_kernel,
            np.array([lengthscale]),
            X[0:1, :],
            X2[0:1, :],
        )
