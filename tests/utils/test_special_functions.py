from math import comb, log, tanh

import lab as B
import numpy as np
import pytest
from sklearn.metrics.pairwise import rbf_kernel

from geometric_kernels.spaces import Hypercube
from geometric_kernels.utils.special_functions import (
    hypercube_heat_kernel,
    kravchuk_normalized,
    walsh_function,
)
from geometric_kernels.utils.utils import (
    binary_vectors_and_subsets,
    check_function_with_backend,
    hamming_distance,
)


@pytest.fixture(params=[1, 2, 3, 5, 10])
def all_xs_and_combs(request):
    """
    Returns a tuple (d, x, combs) where:
    - d is an integer equal to request.param,
    - x is a 2**d x d boolean matrix with all possible binary vectors of length d,
    - combs is a list of all possible combinations of indices of x.
    """
    d = request.param

    x, combs = binary_vectors_and_subsets(d)

    return d, x, combs


def walsh_matrix(d, combs, x):
    return B.stack(*[walsh_function(d, comb, x) for comb in combs])


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_walsh_functions(all_xs_and_combs, backend):
    d, x, combs = all_xs_and_combs

    # Check that Walsh functions are orthogonal.
    check_function_with_backend(
        backend,
        2**d * np.eye(2**d),
        lambda x: B.matmul(walsh_matrix(d, combs, x), B.T(walsh_matrix(d, combs, x))),
        x,
    )

    # Check that Walsh functions only take values in the set {-1, 1}.
    check_function_with_backend(
        backend, np.ones((2**d, 2**d)), lambda x: B.abs(walsh_matrix(d, combs, x)), x
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_kravchuk_polynomials(all_xs_and_combs, backend):
    d, x, combs = all_xs_and_combs

    x0 = np.zeros((1, d), dtype=bool)

    cur_ind = 0
    for j in range(d + 1):
        num_walsh = comb(d, j)

        result = np.sum(
            walsh_matrix(d, combs, x)[cur_ind : cur_ind + num_walsh, :],
            axis=0,
            keepdims=True,
        )

        # Checks that Kravchuk polynomials coincide with certain sums of
        # the Walsh functions.
        check_function_with_backend(
            backend,
            result,
            lambda x0, x: comb(d, j)
            * kravchuk_normalized(d, j, hamming_distance(x0, x)),
            x0,
            x,
        )

        cur_ind += num_walsh


@pytest.mark.parametrize("d", [1, 5, 10])
@pytest.mark.parametrize("lengthscale", [1.0, 5.0, 10.0])
@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_hypercube_heat_kernel(d, lengthscale, backend):
    space = Hypercube(d)

    key = np.random.RandomState()
    N, N2 = key.randint(low=1, high=min(2**d, 10) + 1, size=2)
    key, X = space.random(key, N)
    key, X2 = space.random(key, N2)

    gamma = -log(tanh(lengthscale**2 / 2))
    result = rbf_kernel(X, X2, gamma=gamma)

    # Checks that the heat kernel on the hypercube coincides with the RBF kernel
    # restricted onto binary vectors, with appropriately redefined length scale.
    check_function_with_backend(
        backend,
        result,
        lambda lengthscale, X, X2: hypercube_heat_kernel(
            lengthscale, X, X2, normalized_laplacian=False
        ),
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

        K_first = hypercube_heat_kernel(
            np.array([lengthscale]), X_first, X2_first, normalized_laplacian=False
        )
        K_second = hypercube_heat_kernel(
            np.array([lengthscale]), X_second, X2_second, normalized_laplacian=False
        )

        result = K_first * K_second

        # Checks that the heat kernel of the product is equal to the product
        # of heat kernels.
        check_function_with_backend(
            backend,
            result,
            lambda lengthscale, X, X2: hypercube_heat_kernel(
                lengthscale, X, X2, normalized_laplacian=False
            ),
            np.array([lengthscale]),
            X[0:1, :],
            X2[0:1, :],
        )
