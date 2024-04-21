import lab as B
import numpy as np
import pytest
import tensorflow as tf
import torch
from opt_einsum import contract as einsum
from plum import Tuple

from geometric_kernels.kernels import MaternKarhunenLoeveKernel
from geometric_kernels.lab_extras import from_numpy
from geometric_kernels.spaces.circle import Circle, SinCosEigenfunctions
from geometric_kernels.spaces.eigenfunctions import EigenfunctionsWithAdditionTheorem
from geometric_kernels.utils.utils import chain


class Consts:
    seed = 42
    num_data = 7
    num_data2 = 5
    num_eigenfunctions = 11
    num_levels = 6


def to_typed_tensor(value, backend):
    if backend == "tensorflow":
        return tf.convert_to_tensor(value)
    elif backend == "torch":
        return torch.tensor(value)
    elif backend == "numpy":
        return value
    else:
        raise ValueError("Unknown backend: {}".format(backend))


@pytest.fixture(name="inputs", params=["tensorflow", "torch", "numpy"])
def _inputs_fixure(request) -> Tuple[B.Numeric]:
    np.random.seed(Consts.seed)
    value = np.random.uniform(0, 2 * np.pi, size=(Consts.num_data, 1))
    value2 = np.random.uniform(0, 2 * np.pi, size=(Consts.num_data2, 1))
    return to_typed_tensor(value, request.param), to_typed_tensor(value2, request.param)


@pytest.fixture(name="eigenfunctions")
def _eigenfunctions_fixture():
    eigenfunctions = SinCosEigenfunctions(Consts.num_levels)
    return eigenfunctions


def test_call_eigenfunctions(
    inputs: Tuple[B.Numeric, B.Numeric],
    eigenfunctions: EigenfunctionsWithAdditionTheorem,
):
    inputs, _ = inputs
    output = B.to_numpy(eigenfunctions(inputs))
    assert output.shape == (Consts.num_data, eigenfunctions.num_eigenfunctions)


def test_eigenfunctions_shape(eigenfunctions: EigenfunctionsWithAdditionTheorem):
    num_eigenfunctions_manual = np.sum(eigenfunctions.num_eigenfunctions_per_level)
    assert num_eigenfunctions_manual == eigenfunctions.num_eigenfunctions
    assert len(eigenfunctions.num_eigenfunctions_per_level) == eigenfunctions.num_levels


def test_orthonormality(eigenfunctions: EigenfunctionsWithAdditionTheorem):
    theta = np.linspace(-np.pi, np.pi, 5_000).reshape(-1, 1)  # [N, 1]
    phi = B.to_numpy(eigenfunctions(theta))
    phiT_phi = (phi.T @ phi) * 2 * np.pi / phi.shape[0]
    circumference_circle = 2 * np.pi
    inner_prod = phiT_phi / circumference_circle
    np.testing.assert_array_almost_equal(
        inner_prod, np.eye(inner_prod.shape[0]), decimal=3
    )


def test_weighted_outerproduct_with_addition_theorem(
    inputs, eigenfunctions: EigenfunctionsWithAdditionTheorem
):
    """
    Eigenfunction will use addition theorem to compute outerproduct. We compare against the
    naive implementation.
    """
    inputs, inputs2 = inputs
    weights_per_level = from_numpy(inputs, np.random.randn(eigenfunctions.num_levels))
    weights = B.reshape(weights_per_level, -1, 1)
    chained_weights = chain(
        weights_per_level, eigenfunctions.num_eigenfunctions_per_level
    )
    actual = B.to_numpy(eigenfunctions.weighted_outerproduct(weights, inputs, inputs2))

    Phi_X = eigenfunctions(inputs)
    Phi_X2 = eigenfunctions(inputs2)
    expected = einsum("ni,ki,i->nk", Phi_X, Phi_X2, chained_weights)
    np.testing.assert_array_almost_equal(actual, expected)


def test_weighted_outerproduct_with_addition_theorem_same_input(
    inputs, eigenfunctions: EigenfunctionsWithAdditionTheorem
):
    """
    Eigenfunction will use addition theorem to compute outerproduct. We compare against the
    naive implementation.
    """
    inputs, _ = inputs
    weights_per_level = from_numpy(inputs, np.random.randn(eigenfunctions.num_levels))
    weights = B.reshape(weights_per_level, -1, 1)
    first = B.to_numpy(eigenfunctions.weighted_outerproduct(weights, inputs, inputs))
    second = B.to_numpy(eigenfunctions.weighted_outerproduct(weights, inputs, None))
    np.testing.assert_array_almost_equal(first, second)


def test_weighted_outerproduct_diag_with_addition_theorem(
    inputs, eigenfunctions: EigenfunctionsWithAdditionTheorem
):
    """
    Eigenfunction will use addition theorem to compute outerproduct. We compare against the
    naive implementation.
    """
    inputs, _ = inputs
    weights_per_level = from_numpy(inputs, np.random.randn(eigenfunctions.num_levels))
    chained_weights = chain(
        weights_per_level, eigenfunctions.num_eigenfunctions_per_level
    )
    weights = B.reshape(weights_per_level, -1, 1)
    actual = eigenfunctions.weighted_outerproduct_diag(weights, inputs)

    Phi_X = eigenfunctions(inputs)
    expected = einsum("ni,i->n", Phi_X**2, chained_weights)
    np.testing.assert_array_almost_equal(B.to_numpy(actual), B.to_numpy(expected))


def analytic_kernel(nu: float, r: B.Numeric) -> B.Numeric:
    """
    Analytic implementations of matern-family kernels.

    :param nu: selects the matern
    :param r: distance, shape [...]
    :return: k(r), shape [...]
    """
    r = B.abs(r)
    if nu == 0.5:
        return B.exp(-r)
    elif nu == 1.5:
        sqrt3 = np.sqrt(3.0)
        return (1.0 + sqrt3 * r) * B.exp(-sqrt3 * r)
    elif nu == 2.5:
        sqrt5 = np.sqrt(5.0)
        return (1.0 + sqrt5 * r + 5.0 / 3.0 * (r**2)) * B.exp(-sqrt5 * r)
    elif nu == np.inf:
        return B.exp(-0.5 * r**2)
    else:
        raise NotImplementedError


@pytest.mark.parametrize("nu, decimal", [(0.5, 1), (1.5, 3), (2.5, 5), (np.inf, 6)])
def test_equivalence_kernel(nu, decimal, inputs):
    inputs, inputs2 = inputs
    # Spectral kernel
    circle = Circle()
    kernel = MaternKarhunenLoeveKernel(circle, num_levels=101)
    params = kernel.init_params()
    params["nu"] = from_numpy(inputs, np.r_[nu])
    params["lengthscale"] = from_numpy(inputs, np.r_[1.0])

    K_actual = B.to_numpy(kernel.K(params, inputs, inputs2))

    # Kernel by summing over all distances
    geodesic = inputs[:, None, :] - inputs2[None, :, :]  # [N, N2, 1]
    all_distances = (
        geodesic + np.array([i * 2 * np.pi for i in range(-10, 10)])[None, None, :]
    )
    K_expected = B.to_numpy(B.sum(analytic_kernel(nu, all_distances), axis=2))

    # test equivalence
    np.testing.assert_array_almost_equal(
        K_expected / K_expected[0, 0], K_actual / K_actual[0, 0], decimal=decimal
    )
