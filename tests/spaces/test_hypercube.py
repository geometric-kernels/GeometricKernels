"""
In `test_call_eigenfunctions` we only check the the eigenfunction evaluation
runs, returns the right type, and the shape of the result is correct. The fact
that the values are correct should follow from other tests: that 
"""

import lab as B
import numpy as np
import pytest
from opt_einsum import contract as einsum
from plum import Tuple

from geometric_kernels.kernels import MaternGeometricKernel
from geometric_kernels.spaces import Hypercube
from geometric_kernels.utils.special_functions import hypercube_heat_kernel
from geometric_kernels.utils.utils import (
    binary_vectors_and_subsets,
    chain,
    check_function_with_backend,
)


@pytest.fixture(params=[1, 2, 3, 5, 10])
def inputs(request) -> Tuple[B.Numeric]:
    d = request.param
    space = Hypercube(d)
    key = np.random.RandomState()
    N, N2 = key.randint(low=1, high=min(2**d, 10) + 1, size=2)
    key, X = space.random(key, N)
    key, X2 = space.random(key, N2)
    num_levels = min(space.dim + 1, 5)
    eigenfunctions = space.get_eigenfunctions(num_levels)
    return space, eigenfunctions, X, X2


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_call_eigenfunctions(inputs: Tuple[B.NPNumeric, B.NPNumeric], backend):
    _, eigenfunctions, X, _ = inputs

    check_function_with_backend(
        backend,
        (X.shape[0], eigenfunctions.num_eigenfunctions),
        lambda x: eigenfunctions(x),
        X,
        compare_to_result=lambda res, f_out: f_out.shape == res,
    )


def test_numbers_of_eigenfunctions(inputs):
    space, eigenfunctions, _, _ = inputs
    num_levels = eigenfunctions.num_levels
    assert len(eigenfunctions.num_eigenfunctions_per_level) == num_levels
    assert eigenfunctions.num_eigenfunctions_per_level[0] == 1
    if num_levels == space.dim + 1:
        assert eigenfunctions.num_eigenfunctions == 2**space.dim

    for i in range(num_levels):
        assert eigenfunctions.num_eigenfunctions_per_level[i] > 0

    num_eigenfunctions_manual = sum(eigenfunctions.num_eigenfunctions_per_level)
    assert num_eigenfunctions_manual == eigenfunctions.num_eigenfunctions


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_orthonormality(inputs, backend):
    space, _, _, _ = inputs

    if space.dim > 5:
        pytest.skip("Test is too slow for dim > 5")

    eigenfunctions = space.get_eigenfunctions(space.dim + 1)

    input, _ = binary_vectors_and_subsets(space.dim)

    # Eigenfunctions should be orthonormal with respect to the inner product
    # <x, y> = 2**(-d) sum_i x_i y_i.
    check_function_with_backend(
        backend,
        np.eye(2**space.dim) * 2**space.dim,
        lambda x: B.matmul(B.T(eigenfunctions(x)), eigenfunctions(x)),
        input,
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_weighted_outerproduct_with_addition_theorem(inputs, backend):
    _, eigenfunctions, X, X2 = inputs
    num_levels = eigenfunctions.num_levels

    weights = np.random.rand(num_levels, 1)
    chained_weights = chain(
        weights.squeeze(), eigenfunctions.num_eigenfunctions_per_level
    )

    Phi_X = eigenfunctions(X)
    Phi_X2 = eigenfunctions(X2)
    result = einsum("ni,ki,i->nk", Phi_X, Phi_X2, chained_weights)

    check_function_with_backend(
        backend, result, eigenfunctions.weighted_outerproduct, weights, X, X2
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_weighted_outerproduct_with_addition_theorem_one_input(inputs, backend):
    _, eigenfunctions, X, _ = inputs
    num_levels = eigenfunctions.num_levels

    weights = np.random.rand(num_levels, 1)

    result = eigenfunctions.weighted_outerproduct(weights, X, X)

    check_function_with_backend(
        backend,
        result,
        eigenfunctions.weighted_outerproduct,
        weights,
        X,
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_weighted_outerproduct_diag(inputs, backend):
    _, eigenfunctions, X, _ = inputs
    num_levels = eigenfunctions.num_levels

    weights = np.random.rand(num_levels, 1)

    result = np.diag(eigenfunctions.weighted_outerproduct(weights, X, X))

    check_function_with_backend(
        backend,
        result,
        eigenfunctions.weighted_outerproduct_diag,
        weights,
        X,
    )


@pytest.mark.parametrize("lengthscale", [1.0, 5.0, 10.0])
@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_against_analytic_heat_kernel(inputs, lengthscale, backend):
    space, _, X, X2 = inputs
    lengthscale = np.array([lengthscale])
    result = hypercube_heat_kernel(lengthscale, X, X2)

    kernel = MaternGeometricKernel(space)

    check_function_with_backend(
        backend,
        result,
        lambda nu, lengthscale, X, X2: kernel.K(
            {"nu": nu, "lengthscale": lengthscale}, X, X2
        ),
        np.array([np.inf]),
        lengthscale,
        X,
        X2,
        atol=1e-2,
    )
