from geometric_kernels.kernels.geometric_kernels import MaternKarhunenLoeveKernel
import numpy as np
import pytest
import tensorflow as tf
import gpflow

from geometric_kernels.eigenfunctions import EigenfunctionWithAdditionTheorem
from geometric_kernels.spaces.circle import Circle, SinCosEigenfunctions, cartesian_to_polar
from geometric_kernels.types import TensorLike
from geometric_kernels.utils import chain, l2norm


class Consts:
    num_data = 13
    num_data2 = 7
    num_eigenfunctions = 11


@pytest.fixture(name="inputs")
def _inputs_fixure():
    X = np.random.randn(Consts.num_data, 2)
    X = X / l2norm(X)
    return X

@pytest.fixture(name="inputs2")
def _inputs2_fixure():
    X = np.random.randn(Consts.num_data2, 2)
    X = X / l2norm(X)
    return X


@pytest.fixture(name="eigenfunctions")
def _eigenfunctions_fixture():
    eigenfunctions = SinCosEigenfunctions(Consts.num_eigenfunctions)
    return eigenfunctions


def test_call_eigenfunctions(inputs: TensorLike, eigenfunctions: EigenfunctionWithAdditionTheorem):
    output = eigenfunctions(inputs)
    assert output.numpy().shape == (Consts.num_data, eigenfunctions.num_eigenfunctions)


def test_eigenfunctions_shape(eigenfunctions: EigenfunctionWithAdditionTheorem):
    num_eigenfunctions_manual = np.sum(eigenfunctions.num_eigenfunctions_per_level)
    assert num_eigenfunctions_manual == eigenfunctions.num_eigenfunctions
    assert len(eigenfunctions.num_eigenfunctions_per_level) == eigenfunctions.num_levels


def test_orthonormality(eigenfunctions: EigenfunctionWithAdditionTheorem):
    theta = np.linspace(-np.pi, np.pi, 5_000).reshape(-1, 1)  # [N, 1]
    inputs = np.concatenate([np.cos(theta), np.sin(theta)], axis=-1)  # [N, 2]
    phi = eigenfunctions(inputs).numpy()
    phiT_phi = (phi.T @ phi) * 2 * np.pi / phi.shape[0]
    circumference_circle = 2 * np.pi
    inner_prod = phiT_phi / circumference_circle
    np.testing.assert_array_almost_equal(inner_prod, np.eye(inner_prod.shape[0]), decimal=3)


def test_filter_weights(eigenfunctions: EigenfunctionWithAdditionTheorem):
    weights_per_level = np.random.randn(eigenfunctions.num_levels)
    weights = chain(weights_per_level, eigenfunctions.num_eigenfunctions_per_level).numpy()
    assert len(weights) == eigenfunctions.num_eigenfunctions
    np.testing.assert_array_equal(
        weights_per_level, eigenfunctions._filter_weights(weights).numpy().flatten()
    )


def test_weighted_outerproduct_with_addition_theorem(inputs, inputs2, eigenfunctions: EigenfunctionWithAdditionTheorem):
    """
    Eigenfunction will use addition theorem to compute outerproduct. We compare against the
    naive implementation.
    """
    weights_per_level = np.random.randn(eigenfunctions.num_levels)
    weights = chain(weights_per_level, eigenfunctions.num_eigenfunctions_per_level).numpy()
    actual = eigenfunctions.weighted_outerproduct(weights, inputs, inputs2).numpy()

    Phi_X = eigenfunctions(inputs)
    Phi_X2 = eigenfunctions(inputs2)
    expected = tf.einsum("ni,ki,i->nk", Phi_X, Phi_X2, weights).numpy()
    np.testing.assert_array_almost_equal(actual, expected)


def test_weighted_outerproduct_with_addition_theorem_same_input(inputs, eigenfunctions: EigenfunctionWithAdditionTheorem):
    weights_per_level = np.random.randn(eigenfunctions.num_levels)
    weights = chain(weights_per_level, eigenfunctions.num_eigenfunctions_per_level).numpy()
    first = eigenfunctions.weighted_outerproduct(weights, inputs, inputs).numpy()
    second = eigenfunctions.weighted_outerproduct(weights, inputs, None).numpy()
    np.testing.assert_array_almost_equal(first, second)


def test_weighted_outerproduct_diag_with_addition_theorem(inputs, eigenfunctions: EigenfunctionWithAdditionTheorem):
    """
    Eigenfunction will use addition theorem to compute outerproduct. We compare against the
    naive implementation.
    """
    weights_per_level = np.random.randn(eigenfunctions.num_levels)
    weights = chain(weights_per_level, eigenfunctions.num_eigenfunctions_per_level).numpy()
    actual = eigenfunctions.weighted_outerproduct_diag(weights, inputs).numpy()

    Phi_X = eigenfunctions(inputs)
    expected = tf.einsum("ni,i->n", Phi_X ** 2, weights).numpy()
    np.testing.assert_array_almost_equal(actual, expected)

def Matern52(r):
    sqrt5 = np.sqrt(5.0)
    return (1.0 + sqrt5 * r + 5.0 / 3.0 * tf.square(r)) * tf.exp(-sqrt5 * r)

def test_equivalence_kernel(inputs, inputs2):
    circle = Circle()
    nu = 2.5
    num_eigenfunctions = 21
    kernel = MaternKarhunenLoeveKernel(circle, nu, num_eigenfunctions)
    inputs2 = inputs
    K_actual = kernel.K(inputs, inputs2, lengthscale=1.0)

    theta = cartesian_to_polar(inputs)
    theta2 = cartesian_to_polar(inputs2)
    xi = theta[:, None, :] - theta2[None, :]  # [N, N2, 1]
    angle_between = tf.math.mod(xi, 2 * np.pi)

    values = [Matern52(angle_between + i * 2 * np.pi) for i in range(11)]
    K_expected = tf.concat(values, axis=2)
    K_expected = tf.reduce_sum(K_expected, axis=2)
    
    np.testing.assert_array_almost_equal(K_expected, K_actual)