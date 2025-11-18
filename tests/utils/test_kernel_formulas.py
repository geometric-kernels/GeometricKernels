from math import log, tanh

import numpy as np
import pytest
from sklearn.metrics.pairwise import rbf_kernel

from geometric_kernels.spaces import HammingGraph, HypercubeGraph
from geometric_kernels.utils.kernel_formulas import (
    hamming_graph_heat_kernel,
    hypercube_graph_heat_kernel,
)

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


@pytest.mark.parametrize("d", [1, 5, 10])
@pytest.mark.parametrize("lengthscale", [1.0, 5.0, 10.0])
@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_hamming_graph_reduces_to_hypercube_when_q_equals_2(d, lengthscale, backend):
    space = HypercubeGraph(d)

    key = np.random.RandomState(0)
    N, N2 = key.randint(low=1, high=min(2**d, 10) + 1, size=2)
    key, X = space.random(key, N)
    key, X2 = space.random(key, N2)

    def heat_kernel_hamming(lengthscale, X, X2):
        return hamming_graph_heat_kernel(
            lengthscale, X, X2, q=2, normalized_laplacian=False
        )

    # Compute reference using hypercube formula
    result = hypercube_graph_heat_kernel(
        np.array([lengthscale]), X, X2, normalized_laplacian=False
    )

    # Check that general Hamming formula with q=2 matches hypercube
    check_function_with_backend(
        backend,
        result,
        heat_kernel_hamming,
        np.array([lengthscale]),
        X,
        X2,
    )


@pytest.mark.parametrize("d", [1, 5, 10])
@pytest.mark.parametrize("q", [2, 5, 7])
@pytest.mark.parametrize("lengthscale", [1.0, 5.0, 10.0])
@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_hamming_graph_heat_kernel(d, q, lengthscale, backend):
    space = HammingGraph(d, q)

    key = np.random.RandomState(0)
    N, N2 = key.randint(low=1, high=min(q**d, 10) + 1, size=2)
    key, X = space.random(key, N)
    key, X2 = space.random(key, N2)

    def to_one_hot(X_cat, q):
        """Convert categorical matrix [N, d] to one-hot [N, d*q]."""
        N, d = X_cat.shape
        X_onehot = np.zeros((N, d * q), dtype=float)
        for i in range(N):
            for j in range(d):
                X_onehot[i, j * q + X_cat[i, j]] = 1.0
        return X_onehot

    X_onehot = to_one_hot(X, q)
    X2_onehot = to_one_hot(X2, q)

    beta = lengthscale**2 / 2
    exp_neg_beta_q = np.exp(-beta * q)
    factor_disagree = (1 - exp_neg_beta_q) / (1 + (q - 1) * exp_neg_beta_q)
    gamma = -np.log(factor_disagree) / 2  # one-hot counts differences twice

    result = rbf_kernel(X_onehot, X2_onehot, gamma=gamma)

    def heat_kernel(lengthscale, X, X2):
        return hamming_graph_heat_kernel(
            lengthscale, X, X2, q=q, normalized_laplacian=False
        )

    # Checks that the heat kernel on the Hamming graph coincides with the RBF
    # restricted onto categorical vectors, with appropriately redefined length scale.
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

        K_first = hamming_graph_heat_kernel(
            np.array([lengthscale]), X_first, X2_first, q=q, normalized_laplacian=False
        )
        K_second = hamming_graph_heat_kernel(
            np.array([lengthscale]),
            X_second,
            X2_second,
            q=q,
            normalized_laplacian=False,
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
