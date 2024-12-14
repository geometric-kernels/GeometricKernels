"""
Note: We don't use `check_function_with_backend` throughout this module because
eigenvalues are always represented by numpy arrays, regardless of the backend
used for other routines.
"""

import numpy as np
import pytest

from geometric_kernels.kernels.matern_kernel import default_num

from ..helper import discrete_spectrum_spaces


@pytest.fixture(
    params=discrete_spectrum_spaces(),
    ids=str,
)
def inputs(request):
    """
    Returns a tuple (space, num_levels, eigenvalues) where:
    - space = request.param,
    - num_levels is the default number of levels for the `space`, if it does not
      exceed 100, and 100 otherwise,
    - eigenvalues = space.get_eigenvalues(num_levels),
    - eps, a small number, a technicality for using `assert_array_less`.
    """
    space = request.param
    num_levels = min(default_num(space), 100)
    eigenvalues = space.get_eigenvalues(num_levels)
    eps = 1e-5

    return space, num_levels, eigenvalues, eps


def test_shape(inputs):
    _, num_levels, eigenvalues, _ = inputs

    # Check that the eigenvalues have appropriate shape.
    assert eigenvalues.shape == (num_levels, 1)


def test_positive(inputs):
    _, _, eigenvalues, eps = inputs

    # Check that the eigenvalues are nonnegative.
    np.testing.assert_array_less(np.zeros_like(eigenvalues), eigenvalues + eps)


def test_ordered(inputs):
    _, _, eigenvalues, eps = inputs

    # Check that the eigenvalues are sorted in ascending order.
    np.testing.assert_array_less(eigenvalues[:-1], eigenvalues[1:] + eps)
