import numpy as np
import pytest
from scipy.stats import ks_2samp, multivariate_t

from geometric_kernels.feature_maps.probability_densities import student_t_sample


@pytest.mark.parametrize("deg_freedom, n", [(2, 3), (5, 5), (42, 10)])
def test_student_t_sample(deg_freedom, n):
    size = (2048,)

    key = np.random.RandomState(0)

    shape = 1.0 * np.eye(n)
    loc = 1.0 * np.zeros((n,))

    _, random_sample = student_t_sample(
        key, loc, shape, np.array([1.0 * deg_freedom]), size
    )

    np_random_sample = multivariate_t.rvs(loc, shape, deg_freedom, size, key)

    v = key.standard_normal(n)
    v = v / np.linalg.norm(v)

    random_proj = np.einsum("ni,i->n", random_sample, v)
    np_random_proj = np.einsum("ni,i->n", np_random_sample, v)

    p_value = 0.05

    test_res = ks_2samp(random_proj, np_random_proj)
    assert test_res.pvalue > p_value
