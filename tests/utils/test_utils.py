import numpy as np
import pytest

from geometric_kernels.utils.utils import (
    binary_vectors_and_subsets,
    hamming_distance,
    log_binomial,
)

from ..helper import check_function_with_backend


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_hamming_distance(backend):

    X = np.array([[1, 0, 1]], dtype=bool)

    X2 = np.array([[0, 0, 1]], dtype=bool)

    # Check that hamming_distance gives the correct results for the given inputs.
    check_function_with_backend(backend, np.array([[1]]), hamming_distance, X, X2)
    check_function_with_backend(backend, np.array([[0]]), hamming_distance, X, X)
    check_function_with_backend(backend, np.array([[0]]), hamming_distance, X2, X2)
    check_function_with_backend(backend, np.array([[1]]), hamming_distance, X2, X)

    X = np.asarray(
        [
            [0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )

    X2 = np.asarray(
        [
            [1, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 1, 0, 1],
        ],
        dtype=bool,
    )

    ham_X_X2 = np.asarray(
        [
            [4, 3, 4],
            [2, 1, 2],
            [1, 2, 3],
            [2, 3, 2],
            [3, 4, 5],
            [3, 2, 3],
            [3, 2, 3],
            [3, 2, 3],
        ],
        dtype=int,
    )

    ham_X_X = np.asarray(
        [
            [0, 2, 3, 4, 1, 1, 1, 1],
            [2, 0, 3, 4, 3, 1, 1, 1],
            [3, 3, 0, 1, 2, 4, 4, 4],
            [4, 4, 1, 0, 3, 5, 5, 5],
            [1, 3, 2, 3, 0, 2, 2, 2],
            [1, 1, 4, 5, 2, 0, 0, 0],
            [1, 1, 4, 5, 2, 0, 0, 0],
            [1, 1, 4, 5, 2, 0, 0, 0],
        ],
        dtype=int,
    )

    ham_X2_X2 = np.asarray(
        [
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0],
        ],
        dtype=int,
    )

    # Check that hamming_distance gives the correct results for more given inputs.
    check_function_with_backend(backend, ham_X_X2, hamming_distance, X, X2)
    check_function_with_backend(backend, ham_X_X, hamming_distance, X, X)
    check_function_with_backend(backend, ham_X2_X2, hamming_distance, X2, X2)
    check_function_with_backend(backend, ham_X_X2.T, hamming_distance, X2, X)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_log_binomial(n):
    for k in range(n + 1):
        # Check that log_binomial gives the same result as the log of the
        # binomial coefficient (as computed through `np.math.comb`).
        assert np.isclose(np.log(np.math.comb(n, k)), log_binomial(n, k), atol=1e-10)


@pytest.mark.parametrize("d", [0, 1, 2, 3, 5, 10])
def test_binary_vectors_and_subsets(d):
    X, subsets = binary_vectors_and_subsets(d)

    # Check the returned values have the correct types.
    assert isinstance(X, np.ndarray)
    assert isinstance(subsets, list)

    # Check the returned values have the correct shapes.
    assert X.shape == (2**d, d)
    assert X.dtype == bool
    assert len(subsets) == 2**d

    # Check that all x[i, :] are different and that they have ones at the
    # positions contained in subsets[i] and only there.
    for i in range(2**d):
        xi_alt = np.zeros(d, dtype=bool)
        assert isinstance(subsets[i], list)
        xi_alt[subsets[i]] = True
        assert np.all(X[i, :] == xi_alt)
        for j in range(i + 1, 2**d):
            assert np.any(X[i, :] != X[j, :])
