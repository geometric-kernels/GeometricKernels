"""Regression tests for the space-specific log-domain computation."""

import importlib
from math import comb

import lab as B
import mpmath as mp
import numpy as np
import pytest

from geometric_kernels.feature_maps import (
    DeterministicFeatureMapCompact,
    RandomPhaseFeatureMapHammingGraph,
)
from geometric_kernels.kernels import MaternGeometricKernel, MaternKarhunenLoeveKernel
from geometric_kernels.kernels.matern_kernel_hamming_graph import (
    MaternKernelHammingGraph,
)
from geometric_kernels.spaces import Circle, HammingGraph, HypercubeGraph

from ..helper import np_to_backend


def params(nu=1.5, lengthscale=0.7, dtype=np.float64):
    return dict(
        nu=np.array([nu], dtype=dtype), lengthscale=np.array([lengthscale], dtype=dtype)
    )


def points(d):
    x = np.zeros((3, d), dtype=int)
    x[1, 0] = 1
    x[2, : d // 2] = 1
    return x


@pytest.mark.parametrize("space", [HypercubeGraph(5), HammingGraph(5, 4)])
@pytest.mark.parametrize("levels", [3, 6])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("nu", [0.5, 2.5, np.inf])
def test_small_kernel_matches_standard(space, levels, normalize, nu):
    kernel = MaternKernelHammingGraph(space, levels, normalize=normalize)
    old = MaternKarhunenLoeveKernel(space, levels, normalize=normalize)
    p = params(nu)
    x = points(5)
    np.testing.assert_allclose(kernel.eigenvalues(p), old.eigenvalues(p), rtol=1e-12)
    np.testing.assert_allclose(kernel.K(p, x), old.K(p, x), atol=1e-12)
    np.testing.assert_allclose(kernel.K_diag(p, x), np.diag(kernel.K(p, x)), atol=1e-12)
    np.testing.assert_allclose(kernel.K(p, x, x[:2]), kernel.K(p, x)[:, :2], atol=1e-12)


@pytest.mark.parametrize("d", [32, 128, 1024])
@pytest.mark.parametrize("q", [2, 4, 20])
def test_high_precision_level_weights(d, q):
    space = HammingGraph(d, q)
    kernel = MaternKernelHammingGraph(space, d + 1)
    p = params(lengthscale=0.1)
    # Independent arbitrary-precision powers and exact integer multiplicities.
    with mp.workdps(100):
        raw = [
            (mp.mpf(300) + mp.mpf(q * j) / (d * (q - 1)))
            ** (-mp.mpf("1.5") - mp.mpf(d) / 2)
            for j in range(d + 1)
        ]
        masses = [w * comb(d, j) * (q - 1) ** j for j, w in enumerate(raw)]
        z = mp.fsum(masses)
        expected = np.array([float(m / z) for m in masses])
        expected_log = np.array([float(mp.log(w / z)) for w in raw])[:, None]
    log_w = kernel.log_eigenvalues(p)
    actual = np.exp(
        log_w[:, 0] + kernel.eigenfunctions.log_num_eigenfunctions_per_level
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(log_w, expected_log, rtol=1e-12, atol=1e-10)
    # No cancellation on the diagonal, even when raw spectral values underflow.
    x = points(d)[:1]
    np.testing.assert_allclose(kernel.K(p, x), [[1]], atol=1e-10)
    if d == 1024:
        assert np.all(
            kernel.spectrum(space.get_eigenvalues(d + 1), p["nu"], p["lengthscale"], d)
            == 0
        )


@pytest.mark.parametrize("d,q,levels", [(32, 2, 12), (128, 4, 15), (1024, 20, 20)])
def test_high_precision_kernel(d, q, levels):
    kernel = MaternKernelHammingGraph(HammingGraph(d, q), levels)
    p = params(lengthscale=0.1)
    x = points(d)
    with mp.workdps(100):
        masses = [
            (mp.mpf(300) + mp.mpf(q * j) / (d * (q - 1)))
            ** (-mp.mpf("1.5") - mp.mpf(d) / 2)
            for j in range(levels)
        ]
        z = mp.fsum(masses[j] * comb(d, j) * (q - 1) ** j for j in range(levels))
        expected = []
        for m in [0, 1, d // 2]:
            terms = []
            for j in range(levels):
                # Direct integer polynomial, independent of the recurrence.
                polynomial = sum(
                    (-1) ** r * (q - 1) ** (j - r) * comb(m, r) * comb(d - m, j - r)
                    for r in range(max(0, j - (d - m)), min(j, m) + 1)
                )
                terms.append(masses[j] * polynomial)
            expected.append(float(mp.fsum(terms) / z))
    actual = kernel.K(p, x[:1], x)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("nu", [1.5, np.inf])
def test_deterministic_features(normalize, nu):
    space = HypercubeGraph(6)
    p = params(nu)
    x = points(6)
    for levels in [3, 7]:
        fmap = DeterministicFeatureMapCompact(space, levels)
        _, features = fmap(x, p, normalize=normalize)
        kernel = MaternKernelHammingGraph(space, levels, normalize=normalize)
        np.testing.assert_allclose(features @ features.T, kernel.K(p, x), atol=1e-12)


@pytest.mark.parametrize("space", [HypercubeGraph(6), HammingGraph(6, 4)])
@pytest.mark.parametrize("normalize", [False, True])
def test_random_features_preserve_values(space, normalize):
    p = params()
    x = points(6)
    fmap = RandomPhaseFeatureMapHammingGraph(space, 5, 11)
    _, phases = space.random(np.random.RandomState(23), 11)
    spectrum = MaternKarhunenLoeveKernel.spectrum(
        space.get_eigenvalues(5), p["nu"], p["lengthscale"], 6
    )
    expected = (
        fmap.eigenfunctions.phi_product(x, phases) * np.sqrt(spectrum.T)
    ).reshape(3, -1)
    if normalize:
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    _, actual = fmap(x, p, key=np.random.RandomState(23), normalize=normalize)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("backend", ["numpy", "torch", "tensorflow", "jax"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_backends(backend, dtype):
    if backend != "numpy":
        importlib.import_module("geometric_kernels." + backend)
    if backend == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)

    def cast(x):
        return np_to_backend(x, backend)

    p = {k: cast(v) for k, v in params(dtype=dtype).items()}
    x = cast(points(32))
    kernel = MaternKernelHammingGraph(HammingGraph(32, 4), 12)
    expected = kernel.K(params(), points(32))
    actual = kernel.K(p, x)
    tol = 1e-5 if dtype == np.float32 else 1e-10
    np.testing.assert_allclose(B.to_numpy(actual), expected, rtol=tol, atol=tol)
    assert B.to_numpy(actual).dtype == dtype
    for nu in [1.5, np.inf]:
        p["nu"] = cast(np.array([nu], dtype=dtype))
        full = MaternKernelHammingGraph(HammingGraph(32, 4), 33)
        np.testing.assert_allclose(B.to_numpy(full.K_diag(p, x)), 1, atol=tol)
        assert np.all(np.isfinite(B.to_numpy(full.K(p, x))))
    # Explicit phase locations avoid backend-specific RNG differences.
    fmap = RandomPhaseFeatureMapHammingGraph(HammingGraph(1024, 20), 24)
    log_spectrum = kernel.log_spectrum(
        cast(np.arange(24, dtype=dtype)[:, None] / 1024),
        p["nu"],
        p["lengthscale"],
        1024,
    )
    large_x = cast(points(1024))
    f = fmap._features_from_log_spectrum(log_spectrum, large_x, large_x, True)
    assert np.all(np.isfinite(B.to_numpy(f)))
    np.testing.assert_allclose(np.sum(B.to_numpy(f) ** 2, axis=1), 1, atol=tol)


@pytest.mark.parametrize("backend", ["torch", "tensorflow", "jax"])
@pytest.mark.parametrize("nu", [1.5, np.inf])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("diagonal_only", [False, True])
def test_gradients(backend, nu, dtype, diagonal_only):
    importlib.import_module("geometric_kernels." + backend)
    x = np_to_backend(points(8), backend)
    kernel = MaternKernelHammingGraph(HammingGraph(8, 4), 6)
    full = MaternKernelHammingGraph(HammingGraph(8, 4), 9)
    fmap = RandomPhaseFeatureMapHammingGraph(kernel.space, kernel.num_levels)

    def objective(theta):
        p = {
            "nu": (
                theta[:1]
                if np.isfinite(nu)
                else B.cast(B.dtype(theta), np_to_backend(np.array([np.inf]), backend))
            ),
            "lengthscale": theta[1:],
        }
        if diagonal_only:
            return B.sum(kernel.K_diag(p, x))
        log_s = kernel.log_spectrum(
            kernel.eigenvalues_laplacian, p["nu"], p["lengthscale"], 8
        )
        f = fmap._features_from_log_spectrum(log_s, x, x, True)
        return B.sum(kernel.K(p, x)) + B.sum(full.K(p, x)) + B.sum(f)

    theta_np = np.array([1.5, 0.7], dtype=dtype)
    if backend == "torch":
        import torch

        theta = torch.tensor(theta_np, requires_grad=True)
        gradient = torch.autograd.grad(objective(theta), theta)[0].detach().numpy()
    elif backend == "tensorflow":
        import tensorflow as tf

        theta = tf.Variable(theta_np)
        with tf.GradientTape() as tape:
            value = objective(theta)
        gradient = tape.gradient(value, theta).numpy()
    else:
        import jax

        jax.config.update("jax_enable_x64", True)
        gradient = np.array(jax.grad(objective)(np_to_backend(theta_np, backend)))
    finite_difference = []
    for i in range(2):
        delta = np.zeros(2)
        delta[i] = 1e-5
        finite_difference.append(
            float(
                B.to_numpy(
                    objective(np_to_backend(theta_np + delta, backend))
                    - objective(np_to_backend(theta_np - delta, backend))
                )
            )
            / 2e-5
        )
    assert np.all(np.isfinite(gradient))
    np.testing.assert_allclose(
        gradient,
        finite_difference,
        rtol=1e-5,
        atol=1e-5 if dtype == np.float32 else 1e-7,
    )


def test_heat_small_lengthscale_and_zero():
    x = points(128)
    kernel = MaternKernelHammingGraph(HammingGraph(128, 20), 129)
    for lengthscale in [0, 1e-12, 0.1]:
        p = params(np.inf, lengthscale)
        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = kernel.K(p, x)
        assert np.all(np.isfinite(matrix))
        np.testing.assert_allclose(np.diag(matrix), 1)
        if lengthscale > 0:
            a = lengthscale**2 * 20 / (2 * 128 * 19)
            factor = -np.expm1(-a) / (1 + 19 * np.exp(-a))
            np.testing.assert_allclose(matrix[0, 1], factor, rtol=1e-12, atol=0)


def test_other_spaces_keep_standard_computation():
    kernel = MaternGeometricKernel(Circle(), num=5)
    assert type(kernel) is MaternKarhunenLoeveKernel
    assert not hasattr(kernel, "log_spectrum")
    assert not hasattr(kernel.eigenfunctions, "log_num_eigenfunctions_per_level")


def test_explicit_feature_weights_do_not_underflow_prematurely():
    space = HypercubeGraph(512)
    fmap = DeterministicFeatureMapCompact(space, 1)
    p = params(lengthscale=0.1)
    x = points(512)
    _, raw = fmap(x, p, normalize=False)
    assert np.all(raw > 0)
    assert np.all(raw**2 == 0)
    _, normalized = fmap(x, p)
    np.testing.assert_array_equal(normalized, np.ones((3, 1)))


@pytest.mark.parametrize("q", [2, 4, 20])
@pytest.mark.parametrize("d", [32, 128, 1024])
def test_large_random_features_finite_spectrum(d, q):
    space = HammingGraph(d, q)
    fmap = RandomPhaseFeatureMapHammingGraph(space, min(24, d + 1), 4)
    _, f = fmap(points(d), params(lengthscale=0.1), key=np.random.RandomState(4))
    assert np.all(np.isfinite(f))
    np.testing.assert_allclose(np.sum(f**2, axis=1), 1, atol=1e-10)


def test_zero_linear_weights():
    for space in [HypercubeGraph(3), HammingGraph(3, 4)]:
        phi = space.get_eigenfunctions(4)
        x = points(3)
        with np.errstate(divide="ignore"):
            np.testing.assert_array_equal(
                phi.weighted_outerproduct(np.zeros((4, 1)), x), np.zeros((3, 3))
            )
            np.testing.assert_array_equal(
                phi.weighted_outerproduct_diag(np.zeros((4, 1)), x), np.zeros(3)
            )


@pytest.mark.parametrize("d", [32, 128, 1024])
@pytest.mark.parametrize("nu", [1.5, np.inf])
def test_binary_hamming_matches_hypercube(d, nu):
    p = params(nu)
    x = points(d)
    hypercube = MaternKernelHammingGraph(HypercubeGraph(d), 24)
    hamming = MaternKernelHammingGraph(HammingGraph(d, 2), 24)
    np.testing.assert_array_equal(hypercube.K(p, x), hamming.K(p, x))
    for space in [HypercubeGraph(d), HammingGraph(d, 2)]:
        np.testing.assert_array_equal(space.get_repeated_eigenvalues(1), [[0.0]])


def test_explicit_generic_eigenfunctions():
    from geometric_kernels.spaces.eigenfunctions import EigenfunctionsFromEigenvectors

    space = HypercubeGraph(1)
    phi = EigenfunctionsFromEigenvectors(np.array([[2.0, 0.0], [1.0, 3.0]]))
    options = dict(eigenvalues_laplacian=space.get_eigenvalues(2), eigenfunctions=phi)
    kernel = MaternKernelHammingGraph(space, 2, **options)
    reference = MaternKarhunenLoeveKernel(space, 2, **options)
    x = np.array([[0], [1]])
    p = params()
    np.testing.assert_allclose(kernel.K(p, x), reference.K(p, x), atol=1e-12)
    np.testing.assert_allclose(kernel.K_diag(p, x), reference.K_diag(p, x), atol=1e-12)


@pytest.mark.parametrize("space", [HypercubeGraph(1024), HammingGraph(1024, 4)])
def test_default_log_random_features(space):
    from geometric_kernels.kernels import default_feature_map

    kernel = MaternGeometricKernel(space, num=12)
    for fmap in [
        default_feature_map(space=space, num=12),
        default_feature_map(kernel=kernel),
    ]:
        assert type(fmap) is RandomPhaseFeatureMapHammingGraph
        assert fmap.num_levels == 12
        assert not hasattr(fmap.eigenfunctions, "_random_phase_features")
    _, fmap = MaternGeometricKernel(space, num=12, return_feature_map=True)
    assert type(fmap) is RandomPhaseFeatureMapHammingGraph


def test_other_feature_map_defaults_unchanged():
    from geometric_kernels.feature_maps import RandomPhaseFeatureMapCompact
    from geometric_kernels.kernels import default_feature_map
    from geometric_kernels.spaces import SpecialOrthogonal

    assert (
        type(default_feature_map(space=Circle(), num=3))
        is DeterministicFeatureMapCompact
    )
    assert (
        type(default_feature_map(space=SpecialOrthogonal(3), num=3))
        is RandomPhaseFeatureMapCompact
    )
    with pytest.raises(ValueError, match="HammingGraph or HypercubeGraph"):
        RandomPhaseFeatureMapHammingGraph(Circle(), 3)


@pytest.mark.parametrize("space", [HypercubeGraph(6), HammingGraph(6, 4)])
def test_normalized_addition_theorem(space):
    phi = space.get_eigenfunctions(5)
    x = points(6)
    actual = phi.phi_product_normalized(x, dtype=np.float64)
    expected = phi.phi_product(x) / np.array(phi.num_eigenfunctions_per_level)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
