import lab as B
import numpy as np

from geometric_kernels.kernels import MaternKarhunenLoeveKernel
from geometric_kernels.lab_extras.extras import from_numpy
from geometric_kernels.spaces import Circle, ProductDiscreteSpectrumSpace

_TRUNCATION_LEVEL = 11 ** 2
_GRID_SIZE = 20


def test_circle_product_kernel():
    circle = Circle()
    product = ProductDiscreteSpectrumSpace(circle, circle, num_eigen=11 ** 2)

    grid = B.linspace(0, 2 * B.pi, _GRID_SIZE)
    ones = B.ones(_GRID_SIZE)
    grid = B.stack(
        grid[:, None] * ones[None, :], grid[None, :] * ones[:, None], axis=-1
    )
    grid_ = B.reshape(grid, _GRID_SIZE ** 2, 2)

    for ls in [0.1, 0.5, 1.0, 2.0, 5.0]:

        kernel = MaternKarhunenLoeveKernel(product, 11 ** 2)
        kernel_single = MaternKarhunenLoeveKernel(circle, 11)

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
