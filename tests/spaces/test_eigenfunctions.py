import jax
import lab as B
import numpy as np
import pytest
from opt_einsum import contract as einsum

from geometric_kernels.kernels.matern_kernel import default_num
from geometric_kernels.spaces import CompactMatrixLieGroup, Hypersphere, Mesh
from geometric_kernels.utils.utils import chain

from ..helper import check_function_with_backend, discrete_spectrum_spaces

jax.config.update("jax_enable_x64", True)  # enable float64 in JAX


@pytest.fixture(
    params=discrete_spectrum_spaces(),
    ids=str,
)
def inputs(request):
    """
    Returns a tuple (space, eigenfunctions, X, X2, weights) where:
    - space = request.param,
    - eigenfunctions = space.get_eigenfunctions(num_levels), with reasonable num_levels
    - X is a random sample of random size from the space,
    - X2 is another random sample of random size from the space,
    - weights is an array of positive numbers of shape (eigenfunctions.num_levels, 1).
    """
    space = request.param
    num_levels = default_num(space)
    if isinstance(space, Hypersphere):
        # For Hypersphere, the maximal number of levels with eigenfunction
        # evaluation support is stored in the num_computed_levels field. We
        # do not use more levels to be able to test eigenfunction evaluation.
        num_levels = min(
            10, num_levels, space.get_eigenfunctions(num_levels).num_computed_levels
        )
    elif isinstance(space, Mesh):
        # We limit the number of levels to 50 to avoid excessive computation
        # and increased numerical error for higher order eigenfunctions.
        num_levels = min(50, num_levels)
    eigenfunctions = space.get_eigenfunctions(num_levels)

    key = np.random.RandomState(0)
    N, N2 = key.randint(low=1, high=100 + 1, size=2)
    key, X = space.random(key, N)
    key, X2 = space.random(key, N2)

    # These weights are used for testing the weighted outerproduct, they
    # should be positive.
    weights = np.random.rand(eigenfunctions.num_levels, 1) ** 2 + 1e-5

    return space, eigenfunctions, X, X2, weights


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_call_eigenfunctions(inputs, backend):
    space, eigenfunctions, X, _, _ = inputs

    if isinstance(space, CompactMatrixLieGroup):
        pytest.skip(
            "CompactMatrixLieGroup subclasses do not currently support eigenfunction evaluation"
        )

    # Check that the eigenfunctions can be called, returning the right type and shape.
    check_function_with_backend(
        backend,
        (X.shape[0], eigenfunctions.num_eigenfunctions),
        eigenfunctions,
        X,
        compare_to_result=lambda res, f_out: f_out.shape == res,
    )


def test_numbers_of_eigenfunctions(inputs):
    _, eigenfunctions, _, _, _ = inputs
    num_levels = eigenfunctions.num_levels
    # Check that the length of the `num_eigenfunctions_per_level` list is correct.
    assert len(eigenfunctions.num_eigenfunctions_per_level) == num_levels
    # Check that the first eigenspace is 1-dimensional.
    assert eigenfunctions.num_eigenfunctions_per_level[0] == 1

    # Check that dimensions of eigenspaces are always positive.
    for i in range(num_levels):
        assert eigenfunctions.num_eigenfunctions_per_level[i] > 0

    num_eigenfunctions_manual = sum(eigenfunctions.num_eigenfunctions_per_level)
    # Check that `num_eigenfunctions_per_level` sum up to the total number of
    # eigenfunctions.
    assert num_eigenfunctions_manual == eigenfunctions.num_eigenfunctions


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_orthonormality(inputs, backend):
    space, eigenfunctions, _, _, _ = inputs

    if isinstance(space, CompactMatrixLieGroup):
        pytest.skip(
            "CompactMatrixLieGroup subclasses do not currently support eigenfunction evaluation"
        )

    key = np.random.RandomState(0)
    key, xs = space.random(key, 10000)

    # Check that the eigenfunctions are orthonormal by comparing a Monte Carlo
    # approximation of the inner product with the identity matrix.
    check_function_with_backend(
        backend,
        np.eye(eigenfunctions.num_eigenfunctions),
        lambda xs: B.matmul(B.T(eigenfunctions(xs)), eigenfunctions(xs)) / xs.shape[0],
        xs,
        atol=0.4,  # very loose, but helps make sure the diagonal is close to 1 while the rest is close to 0
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_weighted_outerproduct_with_addition_theorem(inputs, backend):
    space, eigenfunctions, X, X2, weights = inputs

    if isinstance(space, CompactMatrixLieGroup):
        pytest.skip(
            "CompactMatrixLieGroup subclasses do not currently support eigenfunction evaluation"
        )

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
        backend, result, eigenfunctions.weighted_outerproduct, weights, X, X2, atol=1e-2
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_weighted_outerproduct_with_addition_theorem_one_input(inputs, backend):
    _, eigenfunctions, X, _, weights = inputs

    result = eigenfunctions.weighted_outerproduct(weights, X, X)

    # Check that `weighted_outerproduct`, when given only X (but not X2),
    # uses X2=X.
    check_function_with_backend(
        backend,
        result,
        eigenfunctions.weighted_outerproduct,
        weights,
        X,
        atol=1e-2,
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_weighted_outerproduct_diag(inputs, backend):
    _, eigenfunctions, X, _, weights = inputs

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
    _, eigenfunctions, X, X2, weights = inputs

    sum_phi_phi_for_level = eigenfunctions.phi_product(X, X2)

    result = np.einsum("id,...nki->...nk", weights, sum_phi_phi_for_level)

    # Check that `weighted_outerproduct` returns the weighted sum of `phi_product`.
    check_function_with_backend(
        backend, result, eigenfunctions.weighted_outerproduct, weights, X, X2
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_weighted_outerproduct_diag_against_phi_product(inputs, backend):
    _, eigenfunctions, X, _, weights = inputs

    phi_product_diag = eigenfunctions.phi_product_diag(X)

    result = np.einsum("id,ni->n", weights, phi_product_diag)  # [N,]

    # Check that `weighted_outerproduct_diag` returns the weighted sum
    # of `phi_product_diag`.
    check_function_with_backend(
        backend, result, eigenfunctions.weighted_outerproduct_diag, weights, X
    )
