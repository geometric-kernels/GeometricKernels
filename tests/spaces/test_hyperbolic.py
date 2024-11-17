import lab as B
import numpy as np
import pytest

from geometric_kernels.kernels import MaternGeometricKernel
from geometric_kernels.spaces import Hyperbolic
from geometric_kernels.utils.kernel_formulas import (
    hyperbolic_heat_kernel_even,
    hyperbolic_heat_kernel_odd,
)

from ..helper import check_function_with_backend, create_random_state

# from line_profiler import profile


# @profile
@pytest.mark.parametrize("dim", [2, 3, 5, 7])
@pytest.mark.parametrize("lengthscale", [2.0])
@pytest.mark.parametrize("backend", ["tensorflow"])
# @pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_equivalence_kernel(dim, lengthscale, backend):
    space = Hyperbolic(dim)

    key = np.random.RandomState()
    key, X = space.random(key, 6)
    X2 = X.copy()

    t = lengthscale * lengthscale / 2
    if dim % 2 == 1:
        result = hyperbolic_heat_kernel_odd(dim, t, X, X2)
    else:
        result = hyperbolic_heat_kernel_even(dim, t, X, X2)

    kernel = MaternGeometricKernel(space, key=create_random_state(backend))

    # Check that MaternGeometricKernel on Hyperbolic(dim) with nu=inf coincides
    # with the well-known analytic formula for the heat kernel on the hyperbolic
    # space in odd dimensions and semi-analytic formula in even dimensions.
    # We are checking the equivalence on average, computing the norm between
    # the two covariance matrices.
    check_function_with_backend(
        backend,
        result,
        lambda nu, lengthscale, X, X2: kernel.K(
            {"nu": nu, "lengthscale": lengthscale}, X, X2
        ),
        np.array([np.inf]),
        np.array([lengthscale]),
        X,
        X2,
        compare_to_result=lambda res, f_out: np.linalg.norm(res - B.to_numpy(f_out))
        / np.sqrt(res.shape[0] * res.shape[1])
        < 1e-1,
    )
