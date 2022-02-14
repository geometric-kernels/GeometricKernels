import lab as B
import numpy as np
import pytest
import tensorflow as tf
import torch
from opt_einsum import contract as einsum
from plum import Tuple

from geometric_kernels.spaces.hypersphere import Hypersphere, SphericalHarmonics


class Consts:
    seed = 42
    dimension = 2
    num_data = 7
    num_data2 = 5
    num_eigenfunctions = 9


@pytest.fixture(name="eigenfunctions")
def _eigenfunctions_fixture():
    return SphericalHarmonics(Consts.dimension, Consts.num_eigenfunctions)


@pytest.fixture(name="inputs")
def _inputs_fixure(request) -> Tuple[B.Numeric]:
    np.random.seed(Consts.seed)
    value = np.random.randn(Consts.num_data, Consts.dimension + 1)
    value = value / np.sum(value ** 2, axis=-1, keepdims=True)
    value2 = np.random.uniform(Consts.num_data2, Consts.dimension + 1)
    value2 = value2 / np.sum(value2 ** 2, axis=-1, keepdims=True)
    return value, value2


@pytest.mark.parametrize(
    "dim, num, expected_num, expected_num_levels",
    [(2, 9, 9, 3), (2, 10, 16, 4), (3, 14, 14, 3), (3, 15, 30, 4), (8, 9, 10, 2)],
)
def test_shape_eigenfunctions(dim, num, expected_num, expected_num_levels):
    sph_harmonics = SphericalHarmonics(dim, num)
    assert len(sph_harmonics._spherical_harmonics) == sph_harmonics._num_eigenfunctions
    assert sph_harmonics.num_eigenfunctions == expected_num
    assert sph_harmonics.num_levels == expected_num_levels


def test_call_eigenfunctions(
    inputs: Tuple[B.Numeric, B.Numeric],
    eigenfunctions: SphericalHarmonics,
):
    inputs, _ = inputs
    output = eigenfunctions(inputs)
    assert output.shape == (Consts.num_data, eigenfunctions.num_eigenfunctions)


def test_eigenfunctions_shape(eigenfunctions: SphericalHarmonics):
    num_eigenfunctions_manual = np.sum(eigenfunctions.num_eigenfunctions_per_level)
    assert num_eigenfunctions_manual == eigenfunctions.num_eigenfunctions
    assert len(eigenfunctions.num_eigenfunctions_per_level) == eigenfunctions.num_levels


# def test_orthonormality(eigenfunctions: EigenfunctionWithAdditionTheorem):
#     theta = np.linspace(-np.pi, np.pi, 5_000).reshape(-1, 1)  # [N, 1]
#     phi = B.to_numpy(eigenfunctions(theta))
#     phiT_phi = (phi.T @ phi) * 2 * np.pi / phi.shape[0]
#     circumference_circle = 2 * np.pi
#     inner_prod = phiT_phi / circumference_circle
#     np.testing.assert_array_almost_equal(
#         inner_prod, np.eye(inner_prod.shape[0]), decimal=3
#     )


# def test_filter_weights(eigenfunctions: EigenfunctionWithAdditionTheorem):
#     weights_per_level = np.random.randn(eigenfunctions.num_levels)
#     weights = chain(weights_per_level, eigenfunctions.num_eigenfunctions_per_level)
#     assert len(B.to_numpy(weights)) == eigenfunctions.num_eigenfunctions
#     np.testing.assert_array_equal(
#         weights_per_level, B.to_numpy(eigenfunctions._filter_weights(weights)).flatten()
#     )


# def test_weighted_outerproduct_with_addition_theorem(
#     inputs, eigenfunctions: EigenfunctionWithAdditionTheorem
# ):
#     """
#     Eigenfunction will use addition theorem to compute outerproduct. We compare against the
#     naive implementation.
#     """
#     inputs, inputs2 = inputs
#     weights_per_level = from_numpy(inputs, np.random.randn(eigenfunctions.num_levels))
#     weights = chain(weights_per_level, eigenfunctions.num_eigenfunctions_per_level)
#     actual = B.to_numpy(eigenfunctions.weighted_outerproduct(weights, inputs, inputs2))

#     Phi_X = eigenfunctions(inputs)
#     Phi_X2 = eigenfunctions(inputs2)
#     expected = einsum("ni,ki,i->nk", Phi_X, Phi_X2, weights)
#     np.testing.assert_array_almost_equal(actual, expected)


# def test_weighted_outerproduct_with_addition_theorem_same_input(
#     inputs, eigenfunctions: EigenfunctionWithAdditionTheorem
# ):
#     """
#     Eigenfunction will use addition theorem to compute outerproduct. We compare against the
#     naive implementation.
#     """
#     inputs, _ = inputs
#     weights_per_level = from_numpy(inputs, np.random.randn(eigenfunctions.num_levels))
#     weights = chain(weights_per_level, eigenfunctions.num_eigenfunctions_per_level)
#     first = B.to_numpy(eigenfunctions.weighted_outerproduct(weights, inputs, inputs))
#     second = B.to_numpy(eigenfunctions.weighted_outerproduct(weights, inputs, None))
#     np.testing.assert_array_almost_equal(first, second)


# def test_weighted_outerproduct_diag_with_addition_theorem(
#     inputs, eigenfunctions: EigenfunctionWithAdditionTheorem
# ):
#     """
#     Eigenfunction will use addition theorem to compute outerproduct. We compare against the
#     naive implementation.
#     """
#     inputs, _ = inputs
#     weights_per_level = from_numpy(inputs, np.random.randn(eigenfunctions.num_levels))
#     weights = chain(weights_per_level, eigenfunctions.num_eigenfunctions_per_level)
#     actual = eigenfunctions.weighted_outerproduct_diag(weights, inputs)

#     Phi_X = eigenfunctions(inputs)
#     expected = einsum("ni,i->n", Phi_X ** 2, weights)
#     np.testing.assert_array_almost_equal(B.to_numpy(actual), B.to_numpy(expected))


# def analytic_kernel(nu: float, r: B.Numeric) -> B.Numeric:
#     """
#     Analytic implementations of matern-family kernels.

#     :param nu: selects the matern
#     :param r: distance, shape [...]
#     :return: k(r), shape [...]
#     """
#     r = B.abs(r)
#     if nu == 0.5:
#         return B.exp(-r)
#     elif nu == 1.5:
#         sqrt3 = np.sqrt(3.0)
#         return (1.0 + sqrt3 * r) * B.exp(-sqrt3 * r)
#     elif nu == 2.5:
#         sqrt5 = np.sqrt(5.0)
#         return (1.0 + sqrt5 * r + 5.0 / 3.0 * (r ** 2)) * B.exp(-sqrt5 * r)
#     elif nu == np.inf:
#         return B.exp(-0.5 * r ** 2)
#     else:
#         raise NotImplementedError


# @pytest.mark.parametrize("nu, decimal", [(0.5, 1), (1.5, 3), (2.5, 5), (np.inf, 6)])
# def test_equivalence_kernel(nu, decimal, inputs):
#     inputs, inputs2 = inputs
#     # Spectral kernel
#     circle = Circle()
#     kernel = MaternKarhunenLoeveKernel(circle, num_eigenfunctions=101)
#     params, state = kernel.init_params_and_state()
#     params["nu"] = np.r_[nu]
#     params["lengthscale"] = np.r_[1.0]

#     K_actual = B.to_numpy(kernel.K(params, state, inputs, inputs2))

#     # Kernel by summing over all distances
#     geodesic = inputs[:, None, :] - inputs2[None, :, :]  # [N, N2, 1]
#     all_distances = (
#         geodesic + np.array([i * 2 * np.pi for i in range(-10, 10)])[None, None, :]
#     )
#     K_expected = B.to_numpy(B.sum(analytic_kernel(nu, all_distances), axis=2))

#     # test equivalence
#     np.testing.assert_array_almost_equal(
#         K_expected / K_expected[0, 0], K_actual / K_actual[0, 0], decimal=decimal
#     )
