import numpy as np
import pytest

from geometric_kernels.kernels import MaternGeometricKernel
from geometric_kernels.spaces import (
    Circle,
    ProductDiscreteSpectrumSpace,
    SpecialUnitary,
)
from geometric_kernels.utils.product import make_product

from ..helper import check_function_with_backend

_NUM_LEVELS = 20


@pytest.mark.parametrize(
    "factor1, factor2", [(Circle(), Circle()), (Circle(), SpecialUnitary(2))], ids=str
)
@pytest.mark.parametrize("lengthscale", [0.1, 0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_heat_kernel_is_product_of_heat_kernels(factor1, factor2, lengthscale, backend):
    product = ProductDiscreteSpectrumSpace(
        factor1, factor2, num_levels=_NUM_LEVELS**2, num_levels_per_space=_NUM_LEVELS
    )

    key = np.random.RandomState(0)
    key, xs_factor1 = factor1.random(key, 10)
    key, xs_factor2 = factor2.random(key, 10)

    kernel_product = MaternGeometricKernel(product, num=_NUM_LEVELS**2)
    kernel_factor1 = MaternGeometricKernel(factor1, num=_NUM_LEVELS)
    kernel_factor2 = MaternGeometricKernel(factor2, num=_NUM_LEVELS)

    def K_diff(nu, lengthscale, xs_factor1, xs_factor2):
        params = {"nu": nu, "lengthscale": lengthscale}

        xs_product = make_product([xs_factor1, xs_factor2])

        K_product = kernel_product.K(params, xs_product, xs_product)
        K_factor1 = kernel_factor1.K(params, xs_factor1, xs_factor1)
        K_factor2 = kernel_factor2.K(params, xs_factor2, xs_factor2)

        return K_product - K_factor1 * K_factor2

    # Check that the heat kernel on the ProductDiscreteSpectrumSpace coincides
    # with the product of heat kernels on the factors.
    check_function_with_backend(
        backend,
        np.zeros((xs_factor1.shape[0], xs_factor2.shape[0])),
        K_diff,
        np.array([np.inf]),
        np.array([lengthscale]),
        xs_factor1,
        xs_factor2,
    )
