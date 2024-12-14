from itertools import product

import lab as B
import numpy as np
import pytest

from geometric_kernels.kernels import MaternKarhunenLoeveKernel
from geometric_kernels.kernels.matern_kernel import default_num
from geometric_kernels.spaces import Mesh

from ..helper import check_function_with_backend, discrete_spectrum_spaces

_EPS = 1e-5


@pytest.fixture(
    params=product(discrete_spectrum_spaces(), [True, False]),
    ids=lambda tpl: f"{tpl[0]}{'-normalized' if tpl[1] else ''}",
    scope="module",
)
def inputs(request):
    """
    Returns a tuple (space, num_levels, kernel, X, X2) where:
    - space = request.param[0],
    - num_levels = default_num(space),
    - kernel = MaternKarhunenLoeveKernel(space, num_levels, normalize=request.param[1]),
    - X is a random sample of random size from the space,
    - X2 is another random sample of random size from the space,
    """
    space, normalize = request.param
    num_levels = default_num(space)
    kernel = MaternKarhunenLoeveKernel(space, num_levels, normalize=normalize)

    key = np.random.RandomState(0)
    N, N2 = key.randint(low=1, high=100 + 1, size=2)
    key, X = space.random(key, N)
    key, X2 = space.random(key, N2)

    return space, num_levels, kernel, X, X2


def test_params(inputs):
    _, _, kernel, _, _ = inputs

    params = kernel.init_params()

    assert "lengthscale" in params
    assert params["lengthscale"].shape == (1,)
    assert "nu" in params
    assert params["nu"].shape == (1,)


def test_num_levels(inputs):
    _, num_levels, kernel, _, _ = inputs

    assert kernel.eigenfunctions.num_levels == num_levels


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_eigenvalues_shape(inputs, backend):
    _, num_levels, kernel, _, _ = inputs
    params = kernel.init_params()

    # Check that the eigenvalues have appropriate shape.
    check_function_with_backend(
        backend,
        (num_levels, 1),
        kernel.eigenvalues,
        params,
        compare_to_result=lambda res, f_out: B.shape(f_out) == res,
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_eigenvalues_positive(inputs, backend):
    _, _, kernel, _, _ = inputs
    params = kernel.init_params()

    # Check that the eigenvalues are nonnegative.
    check_function_with_backend(
        backend,
        None,
        kernel.eigenvalues,
        params,
        compare_to_result=lambda _, f_out: np.all(B.to_numpy(f_out) >= 0),
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_eigenvalues_ordered(inputs, backend):
    _, _, kernel, _, _ = inputs
    params = kernel.init_params()

    # Check that the eigenvalues are sorted in descending order.
    check_function_with_backend(
        backend,
        None,
        kernel.eigenvalues,
        params,
        compare_to_result=lambda _, f_out: np.all(
            B.to_numpy(f_out)[:-1] >= B.to_numpy(f_out)[1:] - _EPS
        ),
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_K(inputs, backend):
    _, _, kernel, X, X2 = inputs
    params = kernel.init_params()

    result = kernel.K(params, X, X2)

    assert result.shape == (X.shape[0], X2.shape[0]), "K has incorrect shape"

    if backend != "numpy":
        # Check that kernel.K computed using `backend` coincides with the numpy result.
        check_function_with_backend(
            backend,
            result,
            kernel.K,
            params,
            X,
            X2,
        )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_K_one_param(inputs, backend):
    space, _, kernel, X, _ = inputs
    params = kernel.init_params()

    result = kernel.K(params, X, X)

    # Check that kernel.K(X) coincides with kernel.K(X, X).
    check_function_with_backend(
        backend,
        result,
        kernel.K,
        params,
        X,
        atol=1e-2 if isinstance(space, Mesh) else _EPS,
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_K_diag(inputs, backend):
    space, _, kernel, X, _ = inputs
    params = kernel.init_params()

    result = kernel.K(params, X).diagonal()

    assert result.shape == (X.shape[0],), "The diagonal has incorrect shape"

    # Check that kernel.K_diag coincides with the diagonal of kernel.K.
    check_function_with_backend(
        backend,
        result,
        kernel.K_diag,
        params,
        X,
        atol=1e-2 if isinstance(space, Mesh) else _EPS,
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_normalize(inputs, backend):
    space, _, kernel, _, _ = inputs

    if not kernel.normalize:
        pytest.skip("No need to check normalization for an unnormalized kernel")

    params = kernel.init_params()
    key = np.random.RandomState(0)
    key, X = space.random(
        key, 1000
    )  # we need a large sample to get a good estimate of the mean variance

    def mean_variance(params, X):
        return B.reshape(
            B.mean(kernel.K_diag(params, X), squeeze=False),
            1,
        )  # the reshape shields from a bug in lab present at least up to version 1.6.6

    # Check that the average variance of the kernel is 1.
    check_function_with_backend(
        backend,
        np.array([1.0]),
        mean_variance,
        params,
        X,
        atol=0.2,  # very loose, but helps make sure the result is close to 1
    )
