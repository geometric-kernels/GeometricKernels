"""
Implements the standard formulas for the RBF kernel and some Matérn kernels.

The implementation is provided mainly for testing purposes.
"""

from math import sqrt

import lab as B
from beartype.typing import Optional


def euclidean_matern_12_kernel(
    r: B.Numeric,
    lengthscale: Optional[float] = 1.0,
):
    """
    Analytic formula for the Matérn 1/2 kernel on R^d, as a function of
    distance `r` between inputs.

    :param r:
        A batch of distances, an array of shape [...].
    :param lengthscale:
        The length scale of the kernel, defaults to 1.

    :return:
        The kernel values evaluated at `r`, an array of shape [...].
    """

    assert B.all(r >= 0.0)

    return B.exp(-r / lengthscale)


def euclidean_matern_32_kernel(
    r: B.Numeric,
    lengthscale: Optional[float] = 1.0,
):
    """
    Analytic formula for the Matérn 3/2 kernel on R^d, as a function of
    distance `r` between inputs.

    :param r:
        A batch of distances, an array of shape [...].
    :param lengthscale:
        The length scale of the kernel, defaults to 1.

    :return:
        The kernel values evaluated at `r`, an array of shape [...].
    """

    assert B.all(r >= 0.0)

    sqrt3 = sqrt(3.0)
    r = r / lengthscale
    return (1.0 + sqrt3 * r) * B.exp(-sqrt3 * r)


def euclidean_matern_52_kernel(
    r: B.Numeric,
    lengthscale: Optional[float] = 1.0,
):
    """
    Analytic formula for the Matérn 5/2 kernel on R^d, as a function of
    distance `r` between inputs.

    :param r:
        A batch of distances, an array of shape [...].
    :param lengthscale:
        The length scale of the kernel, defaults to 1.

    :return:
        The kernel values evaluated at `r`, an array of shape [...].
    """

    assert B.all(r >= 0.0)

    sqrt5 = sqrt(5.0)
    r = r / lengthscale
    return (1.0 + sqrt5 * r + 5.0 / 3.0 * (r**2)) * B.exp(-sqrt5 * r)


def euclidean_rbf_kernel(
    r: B.Numeric,
    lengthscale: Optional[float] = 1.0,
):
    """
    Analytic formula for the RBF kernel on R^d, as a function of
    distance `r` between inputs.

    :param r:
        A batch of distances, an array of shape [...].
    :param lengthscale:
        The length scale of the kernel, defaults to 1.

    :return:
        The kernel values evaluated at `r`, an array of shape [...].
    """

    assert B.all(r >= 0.0)

    r = r / lengthscale
    return B.exp(-0.5 * r**2)
