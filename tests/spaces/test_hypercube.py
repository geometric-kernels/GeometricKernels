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
    """
    Returns a tuple (space, eigenfunctions, X, X2) where:
    - space is a Hypercube object with dimension equal to request.param,
    - eigenfunctions is the respective Eigenfunctions object with at most 5 levels,
    - X is a random sample of random size from the space,
    - X2 is another random sample of random size from the space.
    """
    d = request.param
    space = Hypercube(d)
    eigenfunctions = space.get_eigenfunctions(min(space.dim + 1, 5))

    key = np.random.RandomState()
    N, N2 = key.randint(low=1, high=min(2**d, 10) + 1, size=2)
    key, X = space.random(key, N)
    key, X2 = space.random(key, N2)

    return space, eigenfunctions, X, X2


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_call_eigenfunctions(inputs: Tuple[B.NPNumeric, B.NPNumeric], backend):
    _, eigenfunctions, X, _ = inputs

    # Check that the eigenfunctions can be called, returning the right type and shape.
    check_function_with_backend(
        backend,
        (X.shape[0], eigenfunctions.num_eigenfunctions),
        lambda X: eigenfunctions(X),
        X,
        compare_to_result=lambda res, f_out: f_out.shape == res,
    )


def test_numbers_of_eigenfunctions(inputs):
    space, eigenfunctions, _, _ = inputs
    num_levels = eigenfunctions.num_levels
    # Check that the length of the `num_eigenfunctions_per_level` list is correct.
    assert len(eigenfunctions.num_eigenfunctions_per_level) == num_levels
    # Check that the first eigenspace is 1-dimensional.
    assert eigenfunctions.num_eigenfunctions_per_level[0] == 1
    # If the number of levels is maximal, check that the number of
    # eigenfunctions is equal to the number of binary vectors of size `space.dim`.
    if num_levels == space.dim + 1:
        assert eigenfunctions.num_eigenfunctions == 2**space.dim

    # Check that dimensions of eigenspaces are always positive.
    for i in range(num_levels):
        assert eigenfunctions.num_eigenfunctions_per_level[i] > 0

    num_eigenfunctions_manual = sum(eigenfunctions.num_eigenfunctions_per_level)
    # Check that `num_eigenfunctions_per_level` sum up to the total number of
    # eigenfunctions.
    assert num_eigenfunctions_manual == eigenfunctions.num_eigenfunctions


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_orthonormality(inputs, backend):
    space, _, _, _ = inputs

    if space.dim > 5:
        pytest.skip("Test is too slow for dim > 5")

    eigenfunctions = space.get_eigenfunctions(space.dim + 1)

    X, _ = binary_vectors_and_subsets(space.dim)

    # Check that the eigenfunctions are orthonormal with respect to the inner
    # product <x, y> = 2**(-d) sum_i x_i y_i.
    check_function_with_backend(
        backend,
        np.eye(2**space.dim) * 2**space.dim,
        lambda X: B.matmul(B.T(eigenfunctions(X)), eigenfunctions(X)),
        X,
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

    # Check that `weighted_outerproduct`, which is based on the addition theorem,
    # returns the same result as the direct computation involving individual
    # eigenfunctions.
    check_function_with_backend(
        backend, result, eigenfunctions.weighted_outerproduct, weights, X, X2
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_weighted_outerproduct_with_addition_theorem_one_input(inputs, backend):
    _, eigenfunctions, X, _ = inputs
    num_levels = eigenfunctions.num_levels

    weights = np.random.rand(num_levels, 1)

    result = eigenfunctions.weighted_outerproduct(weights, X, X)

    # Check that `weighted_outerproduct`, when given only X (but not X2),
    # uses X2=X.
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

    # Check that `weighted_outerproduct_diag` returns the same result as the
    # diagonal of the full `weighted_outerproduct`.
    check_function_with_backend(
        backend,
        result,
        eigenfunctions.weighted_outerproduct_diag,
        weights,
        X,
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_weighted_outerproduct_against_phi_product(inputs, backend):
    _, eigenfunctions, X, X2 = inputs
    num_levels = eigenfunctions.num_levels

    sum_phi_phi_for_level = eigenfunctions.phi_product(X, X2)

    weights = np.random.rand(num_levels, 1)
    result = B.einsum("id,...nki->...nk", weights, sum_phi_phi_for_level)

    # Check that `weighted_outerproduct`, which for the hypercube has a
    # dedicated implementation, returns the same result as the usual way of
    # computing the `weighted_outerproduct` (based on the `phi_product` method).
    check_function_with_backend(
        backend, result, eigenfunctions.weighted_outerproduct, weights, X, X2
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_weighted_outerproduct_diag_against_phi_product(inputs, backend):
    _, eigenfunctions, X, _ = inputs
    num_levels = eigenfunctions.num_levels

    phi_product_diag = eigenfunctions.phi_product_diag(X)

    weights = np.random.rand(num_levels, 1)
    result = B.einsum("id,ni->n", weights, phi_product_diag)  # [N,]

    # Check that `weighted_outerproduct_diag`, which for the hypercube has a
    # dedicated implementation, returns the same result as the usual way of
    # computing the `weighted_outerproduct_diag` (based on the
    # `phi_product_diag` method).
    check_function_with_backend(
        backend, result, eigenfunctions.weighted_outerproduct_diag, weights, X
    )


@pytest.mark.parametrize("lengthscale", [1.0, 5.0, 10.0])
@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_against_analytic_heat_kernel(inputs, lengthscale, backend):
    space, _, X, X2 = inputs
    lengthscale = np.array([lengthscale])
    result = hypercube_heat_kernel(lengthscale, X, X2)

    kernel = MaternGeometricKernel(space)

    # Check that MaternGeometricKernel on the hypercube with nu=infinity
    # coincides with the closed form expression for the heat kernel on the
    # hypercube.
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
