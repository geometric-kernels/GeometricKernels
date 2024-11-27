import lab as B
import numpy as np
import pytest

from geometric_kernels.kernels import MaternFeatureMapKernel
from geometric_kernels.kernels.matern_kernel import default_feature_map, default_num

from ..helper import (
    check_function_with_backend,
    create_random_state,
    noncompact_symmetric_spaces,
)


@pytest.fixture(
    params=noncompact_symmetric_spaces(),
    ids=str,
    scope="module",
)
def inputs(request):
    """
    Returns a tuple (space, num_features, feature_map, X, X2) where:
    - space = request.param,
    - num_features = default_num(space) or 15, whichever is smaller,
    - feature_map = default_feature_map(space=space, num=num_features),
    - X is a random sample of random size from the space,
    - X2 is another random sample of random size from the space,
    """
    space = request.param
    num_features = min(default_num(space), 15)
    feature_map = default_feature_map(space=space, num=num_features)

    key = np.random.RandomState(0)
    N, N2 = key.randint(low=1, high=100 + 1, size=2)
    key, X = space.random(key, N)
    key, X2 = space.random(key, N2)

    return space, num_features, feature_map, X, X2


@pytest.fixture
def kernel(inputs, backend, normalize=True):
    space, _, feature_map, _, _ = inputs

    key = create_random_state(backend)

    return MaternFeatureMapKernel(space, feature_map, key, normalize=normalize)


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_params(inputs, backend, kernel):
    params = kernel.init_params()

    assert "lengthscale" in params
    assert params["lengthscale"].shape == (1,)
    assert "nu" in params
    assert params["nu"].shape == (1,)


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
@pytest.mark.parametrize("normalize", [True, False], ids=["normalize", "no_normalize"])
def test_K(inputs, backend, normalize, kernel):
    _, _, _, X, X2 = inputs
    params = kernel.init_params()

    # Check that kernel.K runs and the output is a tensor of the right backend and shape.
    check_function_with_backend(
        backend,
        (X.shape[0], X2.shape[0]),
        kernel.K,
        params,
        X,
        X2,
        compare_to_result=lambda res, f_out: B.shape(f_out) == res,
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
@pytest.mark.parametrize("normalize", [True, False], ids=["normalize", "no_normalize"])
def test_K_one_param(inputs, backend, normalize, kernel):
    _, _, _, X, _ = inputs
    params = kernel.init_params()

    # Check that kernel.K(X) coincides with kernel.K(X, X).
    check_function_with_backend(
        backend,
        np.zeros((X.shape[0], X.shape[0])),
        lambda params, X: kernel.K(params, X) - kernel.K(params, X, X),
        params,
        X,
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
@pytest.mark.parametrize("normalize", [True, False], ids=["normalize", "no_normalize"])
def test_K_diag(inputs, backend, normalize, kernel):
    _, _, _, X, _ = inputs
    params = kernel.init_params()

    # Check that kernel.K_diag coincides with the diagonal of kernel.K.
    check_function_with_backend(
        backend,
        np.zeros((X.shape[0],)),
        lambda params, X: kernel.K_diag(params, X) - B.diag(kernel.K(params, X)),
        params,
        X,
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_normalize(inputs, backend, kernel):
    _, _, _, X, _ = inputs

    params = kernel.init_params()

    # Check that the variance of the kernel is constant 1.
    check_function_with_backend(
        backend,
        np.ones((X.shape[0],)),
        kernel.K_diag,
        params,
        X,
    )
