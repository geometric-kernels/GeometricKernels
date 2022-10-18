import lab as B
import numpy as np
from opt_einsum import contract as einsum

from geometric_kernels.kernels import MaternKarhunenLoeveKernel
from geometric_kernels.lab_extras.extras import from_numpy
from geometric_kernels.spaces import Circle, ProductDiscreteSpectrumSpace
from geometric_kernels.utils.utils import chain

_TRUNC_LEVEL = 128
_GRID_SIZE = 3


def test_circle_product_eigenfunctions():
    # assert that the naive method of phi-product calculation
    # gives the same result as the addition theorem based calculation
    product = ProductDiscreteSpectrumSpace(
        Circle(), Circle(), num_eigen=_TRUNC_LEVEL**2
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
    chained_weights = chain(weights, eigenfunctions.dim_of_eigenspaces)
    actual = B.to_numpy(eigenfunctions.weighted_outerproduct(weights, X, X))

    expected = einsum("ni,mi,i->nm", Phi_X, Phi_X2, chained_weights)
    np.testing.assert_array_almost_equal(actual, expected)


def test_circle_product_kernel():
    product = ProductDiscreteSpectrumSpace(
        Circle(), Circle(), num_eigen=_TRUNC_LEVEL**2
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

        params, state = kernel.init_params_and_state()
        params["nu"] = from_numpy(grid_, np.inf)
        params["lengthscale"] = from_numpy(grid, ls)

        params_single, state_single = kernel_single.init_params_and_state()
        params_single["nu"] = from_numpy(grid_, np.inf)
        params_single["lengthscale"] = from_numpy(grid, ls)

        k_xx = kernel.K(params, state, grid_, grid_)
        k_xx = k_xx.reshape(_GRID_SIZE, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE)

        k_xx_single_1 = kernel_single.K(
            params_single, state_single, grid_[..., :1], grid_[..., :1]
        ).reshape(_GRID_SIZE, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE)

        k_xx_single_2 = kernel_single.K(
            params_single, state_single, grid_[..., 1:], grid_[..., 1:]
        ).reshape(_GRID_SIZE, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE)

        k_xx_product = k_xx_single_1 * k_xx_single_2

        np.testing.assert_allclose(
            B.to_numpy(k_xx), B.to_numpy(k_xx_product), atol=1e-08, rtol=1e-05
        )
