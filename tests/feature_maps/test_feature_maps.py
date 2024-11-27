import lab as B
import numpy as np
import pytest

from geometric_kernels.feature_maps import RandomPhaseFeatureMapCompact
from geometric_kernels.kernels import MaternGeometricKernel, default_feature_map
from geometric_kernels.kernels.matern_kernel import default_num
from geometric_kernels.spaces import NoncompactSymmetricSpace
from geometric_kernels.utils.utils import make_deterministic

from ..helper import check_function_with_backend, create_random_state, spaces


@pytest.fixture(
    params=spaces(),
    ids=str,
)
def feature_map_and_friends(request, backend):
    """
    Returns a tuple (feature_map, kernel, space) where:
    - feature_map is the `default_feature_map` of the `kernel`,
    - kernel is the `MaternGeometricKernel` on the `space`, with a reasonably
      small value of `num`,
    - space = request.param,

    `backend` parameter is required to create a random state for the feature
    map, if it requires one.
    """
    space = request.param

    if isinstance(space, NoncompactSymmetricSpace):
        kernel = MaternGeometricKernel(
            space, key=create_random_state(backend), num=min(default_num(space), 100)
        )
    else:
        kernel = MaternGeometricKernel(space, num=min(default_num(space), 3))

    feature_map = default_feature_map(kernel=kernel)
    if isinstance(feature_map, RandomPhaseFeatureMapCompact):
        # RandomPhaseFeatureMapCompact requires a key. Note: normally,
        # RandomPhaseFeatureMapNoncompact, RejectionSamplingFeatureMapHyperbolic,
        # and RejectionSamplingFeatureMapSPD also require a key, but when they
        # are obtained from an already constructed kernel's feature map, the key
        # is already provided and fixed in the similar way as we do just below.
        feature_map = make_deterministic(feature_map, key=create_random_state(backend))

    return feature_map, kernel, space


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_feature_map_approximates_kernel(backend, feature_map_and_friends):
    feature_map, kernel, space = feature_map_and_friends

    params = kernel.init_params()

    key = np.random.RandomState(0)
    key, X = space.random(key, 50)

    def diff_kern_mats(params, X):
        _, embedding = feature_map(X, params)

        kernel_mat = kernel.K(params, X, X)
        kernel_mat_alt = B.matmul(embedding, B.T(embedding))

        return kernel_mat - kernel_mat_alt

    # Check that, approximately, k(X, X) = <phi(X), phi(X)>, where k is the
    # kernel and phi is the feature map.
    check_function_with_backend(
        backend,
        np.zeros((X.shape[0], X.shape[0])),
        diff_kern_mats,
        params,
        X,
        atol=0.1,
    )
