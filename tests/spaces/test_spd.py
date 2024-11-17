import lab as B
import numpy as np
import pytest

from geometric_kernels.kernels import MaternGeometricKernel
from geometric_kernels.spaces import SymmetricPositiveDefiniteMatrices
from geometric_kernels.utils.kernel_formulas import spd_heat_kernel_2x2

from ..helper import check_function_with_backend, create_random_state


@pytest.mark.parametrize("lengthscale", [2.0])
@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_equivalence_kernel(lengthscale, backend):
    space = SymmetricPositiveDefiniteMatrices(2)

    key = np.random.RandomState()
    key, X = space.random(key, 5)
    X2 = X.copy()

    t = lengthscale * lengthscale / 2
    result = spd_heat_kernel_2x2(t, X, X2)

    kernel = MaternGeometricKernel(space, key=create_random_state(backend))

    # Check that MaternGeometricKernel on SymmetricPositiveDefiniteMatrices(2)
    # with nu=inf coincides with the semi-analytic formula from :cite:t:`sawyer1992`.
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
