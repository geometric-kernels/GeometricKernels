import numpy as np
import pytest

from geometric_kernels.kernels import MaternGeometricKernel, ProductGeometricKernel
from geometric_kernels.spaces import Circle, SpecialUnitary
from geometric_kernels.utils.product import make_product

from ..helper import check_function_with_backend


@pytest.mark.parametrize(
    "factor1, factor2", [(Circle(), Circle()), (Circle(), SpecialUnitary(2))], ids=str
)
@pytest.mark.parametrize("nu, lengthscale", [(1 / 2, 2.0), (5 / 2, 1.0), (np.inf, 0.1)])
@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_kernel_is_product_of_heat_kernels(factor1, factor2, nu, lengthscale, backend):
    key = np.random.RandomState(0)
    key, xs_factor1 = factor1.random(key, 10)
    key, xs_factor2 = factor2.random(key, 10)

    kernel_factor1 = MaternGeometricKernel(factor1)
    kernel_factor2 = MaternGeometricKernel(factor2)
    product_kernel = ProductGeometricKernel(kernel_factor1, kernel_factor2)

    def K_diff(nu, lengthscale, xs_factor1, xs_factor2):
        params = {"nu": nu, "lengthscale": lengthscale}

        xs_product = make_product([xs_factor1, xs_factor2])

        K_product = product_kernel.K(params, xs_product, xs_product)
        K_factor1 = kernel_factor1.K(params, xs_factor1, xs_factor1)
        K_factor2 = kernel_factor2.K(params, xs_factor2, xs_factor2)

        return K_product - K_factor1 * K_factor2

    # Check that ProductGeometricKernel without ARD coincides with the product
    # of the respective factor kernels.
    check_function_with_backend(
        backend,
        np.zeros((xs_factor1.shape[0], xs_factor2.shape[0])),
        K_diff,
        np.array([nu]),
        np.array([lengthscale]),
        xs_factor1,
        xs_factor2,
    )
