"""
Implements an alternative formula for the heat kernel on the manifold of
symmetric positive definite matrices by :cite:t:`sawyer1992`.

The implementation is adapted from https://github.com/imbirik/LieStationaryKernels.
Since the resulting approximation
* can fail to be positive semi-definite,
* is very slow,
* and is rather numerically unstable,
it is not recommended to use it in practice. The implementation is provided
mainly for testing purposes.
"""

import lab as B
import numpy as np
import scipy
from beartype.typing import Optional


def _spd_heat_kernel_2x2_base(
    t: float,
    x: B.NPNumeric,
    x2: Optional[B.NPNumeric] = None,
) -> float:
    """
    The semi-analytic formula for the heat kernel on manifold of symmetric
    positive definite matrices 2x2 from :cite:t:`sawyer1992`. The implementation
    is adapted from https://github.com/imbirik/LieStationaryKernels.

    :param t:
        The time parameter, a positive float.
    :param x:
        A single input, an array of shape [2, 2].
    :param x2:
        A single input, an array of shape [2, 2]. If None, defaults to x.

    :return:
        An approximation of the kernel value k(x, x2), a float. The kernel is not
        normalized, i.e. k(x, x) may be an arbitrary (implementation-dependent)
        positive number. For the normalized kernel which can also handle batch
        inputs outputting covariance matrices, use :func:`sawyer_heat_kernel`.
    """
    if x2 is None:
        x2 = x

    assert x.shape == (2, 2)
    assert x2.shape == (2, 2)

    cl_1 = np.linalg.cholesky(x)
    cl_2 = np.linalg.cholesky(x2)
    diff = np.linalg.inv(cl_2) @ cl_1
    _, singular_values, _ = np.linalg.svd(diff)
    # Note: singular values that np.linalg.svd outputs are sorted, the following
    # code relies on this fact.
    H1, H2 = np.log(singular_values[0]), np.log(singular_values[1])
    assert H1 >= H2

    r_H_sq = H1 * H1 + H2 * H2
    alpha = H1 - H2

    # Non-integral part
    result = 1.0
    result *= np.exp(-r_H_sq / (4 * t))

    # Integrand
    def link_function(x):
        if x < 1e-5:
            x = 1e-5
        res = 1.0
        res *= 2 * x + alpha
        res *= np.exp(-x * (x + alpha) / (2 * t))
        res *= pow(np.sinh(x) * np.sinh(x + alpha), -1 / 2)
        return res

    # Evaluating the integral

    # scipy.integrate.quad is much more accurate than np.trapz with
    # b_vals = np.logspace(-3., 1, 1000), at least if we believe
    # that Mathematica's NIntegrate is accurate. Also, you might think that
    # scipy.integrate.quad_vec can be used to compute a whole covariance matrix
    # at once. However, it seems to show terrible accuracy in this case.

    integral, error = scipy.integrate.quad(link_function, 0, np.inf)

    result *= integral

    return result


def spd_heat_kernel_2x2(
    t: float,
    X: B.NPNumeric,
    X2: Optional[B.NPNumeric] = None,
) -> B.NPNumeric:
    """
    The semi-analytic formula for the heat kernel on manifold of symmetric
    positive definite matrices 2x2 from :cite:t:`sawyer1992`, normalized to
    have k(x, x) = 1 for all x. The implementation is adapted from
    https://github.com/imbirik/LieStationaryKernels.

    :param t:
        The time parameter, a positive float.
    :param X:
        A batch of inputs, an array of shape [N, 2, 2].
    :param X2:
        A batch of inputs, an array of shape [N2, 2, 2]. If None, defaults to X.

    :return:
        The kernel matrix, an array of shape [N, N2]. The kernel is normalized,
        i.e. k(x, x) = 1 for all x.
    """

    if X2 is None:
        X2 = X

    normalization = _spd_heat_kernel_2x2_base(t, np.eye(2, 2))

    result = np.zeros((X.shape[0], X2.shape[0]))

    # This is a very inefficient implementation, but it will do for tests. The
    # straightforward vectorization of _sawyer_heat_kernel_base is not possible
    # due to scipy.integrate.quad_vec giving very bad accuracy in this case.
    for i, x in enumerate(X):
        for j, x2 in enumerate(X2):
            result[i, j] = _spd_heat_kernel_2x2_base(t, x, x2) / normalization

    return result
