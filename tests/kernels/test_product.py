import lab as B
import numpy as np
from opt_einsum import contract as einsum

from geometric_kernels.kernels import MaternKarhunenLoeveKernel, ProductGeometricKernel
from geometric_kernels.lab_extras.extras import from_numpy
from geometric_kernels.spaces import (
    Circle,
    Hypersphere,
    ProductDiscreteSpectrumSpace,
    SpecialUnitary,
)
from geometric_kernels.utils.product import make_product
from geometric_kernels.utils.utils import chain

_TRUNC_LEVEL = 128
_GRID_SIZE = 3


def test_circle_product_eigenfunctions():
    # assert that the naive method of phi-product calculation
    # gives the same result as the addition theorem based calculation
    product = ProductDiscreteSpectrumSpace(
        Circle(), Circle(), num_levels=_TRUNC_LEVEL**2
    )

    grid = B.linspace(0, 2 * B.pi, _GRID_SIZE)
    ones = B.ones(_GRID_SIZE)
    grid = B.stack(
        grid[:, None] * ones[None, :], grid[None, :] * ones[:, None], axis=-1
    )
    grid_ = B.reshape(grid, _GRID_SIZE**2, 2)

    X = grid_

    eigenfunctions = product.get_eigenfunctions(_TRUNC_LEVEL**2)

    Phi_X = eigenfunctions(X)  # [GS**2, M]
    Phi_X2 = eigenfunctions(X)

    weights = from_numpy(X, np.random.randn(eigenfunctions.num_levels))
    chained_weights = chain(weights, eigenfunctions.num_eigenfunctions_per_level)
    weights = B.expand_dims(weights, -1)
    actual = B.to_numpy(eigenfunctions.weighted_outerproduct(weights, X, X))

    expected = einsum("ni,mi,i->nm", Phi_X, Phi_X2, chained_weights)
    np.testing.assert_array_almost_equal(actual, expected)


def test_circle_product_kernel():
    product = ProductDiscreteSpectrumSpace(
        Circle(), Circle(), num_levels=_TRUNC_LEVEL**2
    )

    grid = B.linspace(0, 2 * B.pi, _GRID_SIZE)
    ones = B.ones(_GRID_SIZE)
    grid = B.stack(
        grid[:, None] * ones[None, :], grid[None, :] * ones[:, None], axis=-1
    )
    grid_ = B.reshape(grid, _GRID_SIZE**2, 2)

    for ls in [0.1, 0.5, 1.0, 2.0, 5.0]:
        kernel = MaternKarhunenLoeveKernel(product, _TRUNC_LEVEL**2)
        kernel_single = MaternKarhunenLoeveKernel(Circle(), _TRUNC_LEVEL)

        params = kernel.init_params()
        params["nu"] = from_numpy(grid_, [np.inf])
        params["lengthscale"] = from_numpy(grid, [ls])

        params_single = kernel_single.init_params()
        params_single["nu"] = from_numpy(grid_, [np.inf])
        params_single["lengthscale"] = from_numpy(grid, [ls])

        k_xx = kernel.K(params, grid_, grid_)
        k_xx = k_xx.reshape(_GRID_SIZE, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE)

        k_xx_single_1 = kernel_single.K(
            params_single, grid_[..., :1], grid_[..., :1]
        ).reshape(_GRID_SIZE, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE)

        k_xx_single_2 = kernel_single.K(
            params_single, grid_[..., 1:], grid_[..., 1:]
        ).reshape(_GRID_SIZE, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE)

        k_xx_product = k_xx_single_1 * k_xx_single_2

        np.testing.assert_allclose(
            B.to_numpy(k_xx), B.to_numpy(k_xx_product), atol=1e-08, rtol=1e-05
        )


def test_product_space_circle_su():
    circle = Circle()
    su = SpecialUnitary(2)

    product = ProductDiscreteSpectrumSpace(
        circle,
        su,
        num_levels=400,
        num_levels_per_space=20,
    )

    key = B.create_random_state(np.float32)
    key, xs_circle = circle.random(key, 1000)
    key, xs_su = su.random(key, 1000)

    xs = make_product([xs_circle, xs_su])

    kernel = MaternKarhunenLoeveKernel(product, 400)
    kernel_single_circle = MaternKarhunenLoeveKernel(circle, 20)
    kernel_single_su = MaternKarhunenLoeveKernel(su, 20)

    for ls in [0.1, 0.5, 1.0, 2.0, 5.0]:

        params = kernel.init_params()
        params["nu"] = np.r_[np.inf]
        params["lengthscale"] = np.r_[ls]

        k_xx = kernel.K(params, xs, xs[:1])  # [N, 1]

        k_xx_circle = kernel_single_circle.K(params, xs_circle, xs_circle[:1])  # [N, 1]

        k_xx_su = kernel_single_su.K(params, xs_su, xs_su[:1])  # [N, 1]

        k_xx_product = k_xx_circle * k_xx_su

        np.testing.assert_allclose(k_xx, k_xx_product, atol=1e-08, rtol=1e-05)


def test_product_space_circle_su_and_product_kernel():
    circle = Circle()
    su = SpecialUnitary(2)

    product = ProductDiscreteSpectrumSpace(
        circle,
        su,
        num_levels=400,
        num_levels_per_space=20,
    )

    key = B.create_random_state(np.float32)
    key, xs_circle = circle.random(key, 1000)
    key, xs_su = su.random(key, 1000)

    xs = make_product([xs_circle, xs_su])

    kernel = MaternKarhunenLoeveKernel(product, 400)

    kernel_single_circle = MaternKarhunenLoeveKernel(circle, 20)
    kernel_single_su = MaternKarhunenLoeveKernel(su, 20)

    product_kernel = ProductGeometricKernel(kernel_single_circle, kernel_single_su)

    for ls in [0.1, 0.5, 1.0, 2.0, 5.0]:

        params = kernel.init_params()
        params["nu"] = np.r_[np.inf]
        params["lengthscale"] = np.r_[ls]
        product_params = {
            "nu": np.r_[np.inf, np.inf],
            "lengthscale": np.r_[ls, ls],
        }

        k_xx = kernel.K(params, xs, xs[:1])  # [N, 1]
        k_xx_product = product_kernel.K(product_params, xs, xs[:1])  # [N, 1]

        np.testing.assert_allclose(k_xx, k_xx_product, atol=1e-08, rtol=1e-05)


def test_number_of_individual_eigenfunctions():
    circle = Circle()
    sphere = Hypersphere(3)

    product = ProductDiscreteSpectrumSpace(
        circle,
        sphere,
        num_levels=5,
        num_levels_per_space=20,
    )

    eigf = product.get_eigenfunctions(5)

    key = B.create_random_state(np.float32)
    N = 10
    key, xs_circle = circle.random(key, N)
    key, xs_sph = sphere.random(key, N)

    xs = make_product([xs_circle, xs_sph])

    assert eigf(xs).shape == (N, eigf.num_eigenfunctions)
