import numpy as np
import pytest
from scipy.stats import multivariate_t, ks_2samp

from geometric_kernels.feature_maps.probability_densities import student_t_sample


@pytest.mark.parametrize("deg_freedom, n", [(2, 5, 42), (3, 5, 6)])
def test_student_t_sample(deg_freedom):
    size = (1024,)

    key = np.random.RandomState(seed=1234)

    shape = 1.0*np.eye(n)
    loc = 1.0*np.zeros((n,))

    _, random_sample = student_t_sample(key, loc, shape, deg_freedom, size)

    np_random_sample = multivariate_t(loc, shape, deg_freedom, size=size, seed=key)

    p_value = 0.05

    test_res = ks_2samp(random_sample, np_random_sample)
    assert test_res.pvalue > p_value
