import itertools

import lab as B
import numpy as np
import pytest
from numpy.testing import assert_allclose
from opt_einsum import contract as einsum

from geometric_kernels.kernels.feature_maps import random_phase_feature_map_compact
from geometric_kernels.kernels.geometric_kernels import MaternKarhunenLoeveKernel
from geometric_kernels.spaces.su import SUGroup


@pytest.mark.parametrize(
    "group_cls, n, order, dtype",
    [
        # (SOGroup, 3, 10, np.double),
        # (SOGroup, 4, 10, np.double),
        # (SOGroup, 5, 10, np.double),
        # (SOGroup, 6, 10, np.double),
        # (SOGroup, 7, 10, np.double),
        # (SOGroup, 8, 10, np.double),
        (SUGroup, 2, 20, np.cdouble),
        # (SUGroup, 3, 20, np.cdouble),
        # (SUGroup, 4, 20, np.cdouble),
        # (SUGroup, 5, 20, np.cdouble),
        # (SUGroup, 6, 20, np.cdouble),
    ],
)
def test_compact_lie_groups(group_cls, n, order, dtype):
    key = B.create_random_state(dtype, seed=0)

    group = group_cls(n=n)
    eigenfunctions = group.get_eigenfunctions(order)

    kernel = MaternKarhunenLoeveKernel(group, order, normalize=True)
    param = dict(lengthscale=np.array(5), nu=np.array(1.5))

    feature_order = 5000
    feature_map = random_phase_feature_map_compact(group, order, feature_order)

    b1, b2 = 10, 10
    key, x = group.random(key, b1)
    key, y = group.random(key, b2)

    eye_ = np.matmul(x, group.inverse(x))[None, ...]
    diff = eye_ - np.eye(n, dtype=dtype)
    zeros = np.zeros_like(eye_)
    assert_allclose(diff, zeros, atol=1e-5)

    num_samples_x = 20
    num_samples_g = 20
    key, xs = group.random(key, num_samples_x)
    key, gs = group.random(key, num_samples_g)
    conjugates = np.matmul(np.matmul(gs, xs), group.inverse(gs))

    conj_gammas = eigenfunctions._torus_representative(conjugates)
    xs_gammas = eigenfunctions._torus_representative(conjugates)
    for chi in eigenfunctions._characters:
        chi_vals_xs = chi(xs_gammas)
        chi_vals_conj = chi(conj_gammas)
        assert_allclose(chi_vals_xs, chi_vals_conj)

    identity = np.eye(n, dtype=dtype).reshape(1, n, n)
    identity_gammas = eigenfunctions._torus_representative(identity)
    dimensions = eigenfunctions._dimensions
    characters = eigenfunctions._characters
    for chi, dim in zip(characters, dimensions):
        chi_val = chi(identity_gammas)
        assert_allclose(chi_val.real, dim)
        assert_allclose(chi_val.imag, 0)

    num_samples_x = 5 * 10**5
    key, xs = group.random(key, num_samples_x)
    gammas = eigenfunctions._torus_representative(xs)
    characters = eigenfunctions._characters
    scalar_products = np.zeros((order, order), dtype=dtype)
    for a, b in itertools.product(enumerate(characters), repeat=2):
        i, chi1 = a
        j, chi2 = b
        scalar_products[i, j] = np.mean((np.conj(chi1(gammas)) * chi2(gammas)).real)
    print(np.max(np.abs(scalar_products - np.eye(order, dtype=dtype))))
    assert_allclose(scalar_products, np.eye(order, dtype=dtype), atol=5e-2)

    identity = np.eye(n, dtype=dtype).reshape(-1, n, n)

    K_xx = (kernel.K(param, x, x)).real
    key, embed_x = feature_map(x, param, key=key, normalize=True)
    F_xx = (einsum("ni,mi-> nm", embed_x, embed_x.conj())).real
    print(K_xx)
    print("-------")
    print(F_xx)
    assert_allclose(K_xx, F_xx, atol=5e-2)
