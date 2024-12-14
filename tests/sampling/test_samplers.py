import lab as B
import numpy as np
import pytest

from geometric_kernels.sampling import sampler

from ..feature_maps.test_feature_maps import feature_map_and_friends  # noqa: F401
from ..helper import check_function_with_backend, create_random_state

_NUM_SAMPLES = 2


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_output_shape_and_backend(backend, feature_map_and_friends):
    feature_map, kernel, space = feature_map_and_friends

    params = kernel.init_params()
    sample_paths = sampler(feature_map, s=_NUM_SAMPLES)

    key = np.random.RandomState(0)
    key, X = space.random(key, 50)

    def sample(params, X):
        return sample_paths(X, params, key=create_random_state(backend))[1]

    # Check that sample_paths runs and the output is a tensor of the right backend and shape.
    check_function_with_backend(
        backend,
        (X.shape[0], _NUM_SAMPLES),
        sample,
        params,
        X,
        compare_to_result=lambda res, f_out: B.shape(f_out) == res,
    )
