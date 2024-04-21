import itertools

import lab as B
import numpy as np
import pytest
from numpy.testing import assert_allclose
from opt_einsum import contract as einsum

from geometric_kernels.feature_maps import RandomPhaseFeatureMapCompact
from geometric_kernels.kernels import MaternKarhunenLoeveKernel
from geometric_kernels.spaces import SpecialOrthogonal, SpecialUnitary


@pytest.fixture(name="group_cls", params=["so", "su"])
def _group_cls(request):
    if request.param == "so":
        return SpecialOrthogonal
    elif request.param == "su":
        return SpecialUnitary


@pytest.fixture(name="group", params=[3, 5])
def _group(group_cls, request):
    group = group_cls(n=request.param)
    return group


@pytest.fixture(name="group_and_eigf", params=[10])
def _group_and_eigf(group, request):
    eigf = group.get_eigenfunctions(num=request.param)
    return group, eigf


def get_dtype(group):
    if isinstance(group, SpecialOrthogonal):
        return np.double
    elif isinstance(group, SpecialUnitary):
        return np.cdouble
    else:
        raise ValueError()


def test_group_inverse(group_and_eigf):
    group, eigenfunctions = group_and_eigf
    dtype = get_dtype(group)

    key = B.create_random_state(dtype, seed=0)

    b1, b2 = 10, 10
    key, x = group.random(key, b1)
    key, y = group.random(key, b2)

    eye_ = np.matmul(x, group.inverse(x))[None, ...]
    diff = eye_ - np.eye(group.n, dtype=dtype)
    zeros = np.zeros_like(eye_)

    assert_allclose(diff, zeros, atol=1e-5)


def test_character_conj_invariant(group_and_eigf):
    group, eigenfunctions = group_and_eigf
    dtype = get_dtype(group)

    key = B.create_random_state(dtype, seed=0)

    num_samples_x = 20
    num_samples_g = 20
    key, xs = group.random(key, num_samples_x)
    key, gs = group.random(key, num_samples_g)
    conjugates = np.matmul(np.matmul(gs, xs), group.inverse(gs))

    conj_gammas = eigenfunctions._torus_representative(conjugates)
    xs_gammas = eigenfunctions._torus_representative(xs)
    for chi in eigenfunctions._characters:
        chi_vals_xs = chi(xs_gammas)
        chi_vals_conj = chi(conj_gammas)
        assert_allclose(chi_vals_xs, chi_vals_conj)


def test_character_at_identity(group_and_eigf):
    group, eigenfunctions = group_and_eigf
    dtype = get_dtype(group)

    identity = np.eye(group.n, dtype=dtype).reshape(1, group.n, group.n)
    identity_gammas = eigenfunctions._torus_representative(identity)
    dimensions = eigenfunctions._dimensions
    characters = eigenfunctions._characters
    for chi, dim in zip(characters, dimensions):
        chi_val = chi(identity_gammas)
        assert_allclose(chi_val.real, dim)
        assert_allclose(chi_val.imag, 0)


def test_characters_orthogonal(group_and_eigf):
    group, eigenfunctions = group_and_eigf
    dtype = get_dtype(group)
    order = eigenfunctions.num_levels

    key = B.create_random_state(dtype, seed=0)

    num_samples_x = 5 * 10**5
    key, xs = group.random(key, num_samples_x)
    gammas = eigenfunctions._torus_representative(xs)
    characters = eigenfunctions._characters
    scalar_products = np.zeros((order, order), dtype=dtype)
    for a, b in itertools.product(enumerate(characters), repeat=2):
        i, chi1 = a
        j, chi2 = b
        scalar_products[i, j] = np.mean((np.conj(chi1(gammas)) * chi2(gammas)).real)

    assert_allclose(scalar_products, np.eye(order, dtype=dtype), atol=5e-2)


def test_feature_map(group_and_eigf):
    group, eigenfunctions = group_and_eigf
    order = eigenfunctions.num_levels
    dtype = get_dtype(group)
    key = B.create_random_state(dtype, seed=0)

    kernel = MaternKarhunenLoeveKernel(group, order, normalize=True)
    param = dict(lengthscale=np.array([10]), nu=np.array([1.5]))

    feature_order = 5000
    feature_map = RandomPhaseFeatureMapCompact(group, order, feature_order)

    key, x = group.random(key, 10)

    K_xx = (kernel.K(param, x, x)).real
    key, embed_x = feature_map(x, param, key=key, normalize=True)
    F_xx = (einsum("ni,mi-> nm", embed_x, embed_x.conj())).real

    assert_allclose(K_xx, F_xx, atol=5e-2)
