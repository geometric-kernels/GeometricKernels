import lab as B
import numpy as np
import pytest

from geometric_kernels.kernels import MaternGeometricKernel
from geometric_kernels.spaces.circle import Circle
from geometric_kernels.utils.kernel_formulas import (
    euclidean_matern_12_kernel,
    euclidean_matern_32_kernel,
    euclidean_matern_52_kernel,
    euclidean_rbf_kernel,
)

from ..helper import check_function_with_backend


@pytest.mark.parametrize(
    "nu, atol", [(0.5, 1e-1), (1.5, 1e-3), (2.5, 1e-3), (np.inf, 1e-5)]
)
@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_equivalence_kernel(nu, atol, backend):
    if nu == 0.5:
        analytic_kernel = euclidean_matern_12_kernel
    elif nu == 1.5:
        analytic_kernel = euclidean_matern_32_kernel
    elif nu == 2.5:
        analytic_kernel = euclidean_matern_52_kernel
    elif nu == np.inf:
        analytic_kernel = euclidean_rbf_kernel

    inputs = np.random.uniform(0, 2 * np.pi, size=(5, 1))
    inputs2 = np.random.uniform(0, 2 * np.pi, size=(3, 1))

    # Compute kernel using periodic summation
    geodesic = inputs[:, None, :] - inputs2[None, :, :]  # [N, N2, 1]
    all_distances = (
        geodesic + np.array([i * 2 * np.pi for i in range(-10, 10)])[None, None, :]
    )
    all_distances = B.abs(all_distances)
    result = B.to_numpy(B.sum(analytic_kernel(all_distances), axis=2))

    kernel = MaternGeometricKernel(Circle())

    # Check that MaternGeometricKernel on Circle() coincides with the
    # periodic summation of the respective Euclidean Matérn kernel.
    check_function_with_backend(
        backend,
        result,
        kernel.K,
        {"nu": np.array([nu]), "lengthscale": np.array([1.0])},
        inputs,
        inputs2,
        atol=atol,
    )
