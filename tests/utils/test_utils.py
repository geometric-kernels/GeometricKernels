import numpy as np
import pytest

from geometric_kernels.utils.utils import (
    binary_vectors_and_subsets,
    check_function_with_backend,
    hamming_distance,
    log_binomial,
)


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_hamming_distance(backend):

    x1 = np.array([[1, 0, 1]], dtype=bool)

    x2 = np.array([[0, 0, 1]], dtype=bool)

    # Check that hamming_distance gives the correct results for the given inputs.
    check_function_with_backend(backend, np.array([[1]]), hamming_distance, x1, x2)
    check_function_with_backend(backend, np.array([[0]]), hamming_distance, x1, x1)
    check_function_with_backend(backend, np.array([[0]]), hamming_distance, x2, x2)
    check_function_with_backend(backend, np.array([[1]]), hamming_distance, x2, x1)

    x1 = np.asarray(
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

    x2 = np.asarray(
        [
            [1, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 1, 0, 1],
        ],
        dtype=bool,
    )

    ham_x1_x2 = np.asarray(
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

    ham_x1_x1 = np.asarray(
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

    ham_x2_x2 = np.asarray(
        [
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0],
        ],
        dtype=int,
    )

    # Check that hamming_distance gives the correct results for more given inputs.
    check_function_with_backend(backend, ham_x1_x2, hamming_distance, x1, x2)
    check_function_with_backend(backend, ham_x1_x1, hamming_distance, x1, x1)
    check_function_with_backend(backend, ham_x2_x2, hamming_distance, x2, x2)
    check_function_with_backend(backend, ham_x1_x2.T, hamming_distance, x2, x1)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_log_binomial(n):
    for k in range(n + 1):
        # Check that log_binomial gives the same result as the log of the
        # binomial coefficient (as computed through `np.math.comb`).
        assert np.isclose(np.log(np.math.comb(n, k)), log_binomial(n, k), atol=1e-10)


@pytest.mark.parametrize("d", [0, 1, 2, 3, 5, 10])
def test_binary_vectors_and_subsets(d):
    x, subsets = binary_vectors_and_subsets(d)

    # Check the returned values have the correct types.
    assert isinstance(x, np.ndarray)
    assert isinstance(subsets, list)

    # Check the returned values have the correct shapes.
    assert x.shape == (2**d, d)
    assert x.dtype == bool
    assert len(subsets) == 2**d

    # Check that all x[i, :] are different and that they have ones at the
    # positions contained in subsets[i] and only there.
    for i in range(2**d):
        xi_alt = np.zeros(d, dtype=bool)
        assert isinstance(subsets[i], list)
        xi_alt[subsets[i]] = True
        assert np.all(x[i, :] == xi_alt)
        for j in range(i + 1, 2**d):
            assert np.any(x[i, :] != x[j, :])
