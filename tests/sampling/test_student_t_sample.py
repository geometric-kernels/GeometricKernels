import numpy as np
import pytest
import scipy.stats

from geometric_kernels.feature_maps.probability_densities import student_t_sample


@pytest.mark.parametrize("deg_freedom", [2, 5, 42])
def test_student_t_sample(deg_freedom):
    size = (1024,)

    key = np.random.RandomState(seed=1234)
    default_rng = np.random.default_rng(seed=1234)

    _, random_sample = student_t_sample(key, size, np.r_[deg_freedom])

    np_random_sample = default_rng.standard_t(deg_freedom, size=size)

    p_value = 0.05

    test_res = scipy.stats.ks_2samp(random_sample, np_random_sample)
    assert test_res.pvalue > p_value
