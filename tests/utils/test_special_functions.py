from math import comb

import lab as B
import numpy as np
import pytest

from geometric_kernels.utils.special_functions import (
    kravchuk_normalized,
    walsh_function,
)
from geometric_kernels.utils.utils import binary_vectors_and_subsets, hamming_distance

from ..helper import check_function_with_backend


@pytest.fixture(params=[1, 2, 3, 5, 10])
def all_xs_and_combs(request):
    """
    Returns a tuple (d, x, combs) where:
    - d is an integer equal to request.param,
    - x is a 2**d x d boolean matrix with all possible binary vectors of length d,
    - combs is a list of all possible combinations of indices of x.
    """
    d = request.param

    X, combs = binary_vectors_and_subsets(d)

    return d, X, combs


def walsh_matrix(d, combs, X):
    return B.stack(*[walsh_function(d, comb, X) for comb in combs])


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_walsh_functions(all_xs_and_combs, backend):
    d, X, combs = all_xs_and_combs

    # Check that Walsh functions are orthogonal.
    check_function_with_backend(
        backend,
        2**d * np.eye(2**d),
        lambda X: B.matmul(walsh_matrix(d, combs, X), B.T(walsh_matrix(d, combs, X))),
        X,
    )

    # Check that Walsh functions only take values in the set {-1, 1}.
    check_function_with_backend(
        backend, np.ones((2**d, 2**d)), lambda X: B.abs(walsh_matrix(d, combs, X)), X
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_kravchuk_polynomials(all_xs_and_combs, backend):
    d, X, combs = all_xs_and_combs

    x0 = np.zeros((1, d), dtype=bool)

    cur_ind = 0
    for j in range(d + 1):
        num_walsh = comb(d, j)

        result = np.sum(
            walsh_matrix(d, combs, X)[cur_ind : cur_ind + num_walsh, :],
            axis=0,
            keepdims=True,
        )

        def krav(x0, X):
            return comb(d, j) * kravchuk_normalized(d, j, hamming_distance(x0, X))

        # Checks that Kravchuk polynomials coincide with certain sums of
        # the Walsh functions.
        check_function_with_backend(
            backend,
            result,
            krav,
            x0,
            X,
        )

        cur_ind += num_walsh


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_kravchuk_precomputed(all_xs_and_combs, backend):
    d, X, _ = all_xs_and_combs

    x0 = np.zeros((1, d), dtype=bool)

    kravchuk_normalized_j_minus_1, kravchuk_normalized_j_minus_2 = None, None
    for j in range(d + 1):

        cur_kravchuk_normalized = kravchuk_normalized(d, j, hamming_distance(x0, X))

        def krav(x0, X, kn1, kn2):
            return kravchuk_normalized(d, j, hamming_distance(x0, X), kn1, kn2)

        # Checks that Kravchuk polynomials coincide with certain sums of
        # the Walsh functions.
        check_function_with_backend(
            backend,
            cur_kravchuk_normalized,
            krav,
            x0,
            X,
            kravchuk_normalized_j_minus_1,
            kravchuk_normalized_j_minus_2,
        )

        kravchuk_normalized_j_minus_2 = kravchuk_normalized_j_minus_1
        kravchuk_normalized_j_minus_1 = cur_kravchuk_normalized
