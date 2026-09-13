"""
Implements the closed form expression for the heat kernel on the Hamming graph.

The implementation is provided mainly for testing purposes.
"""

from math import sqrt

import lab as B
from beartype.typing import Optional

from geometric_kernels.lab_extras import expm1
from geometric_kernels.utils.utils import _check_1_vector, _check_matrix


def hamming_graph_heat_kernel(
    lengthscale: B.Numeric,
    X: B.Numeric,
    X2: Optional[B.Numeric] = None,
    q: int = 2,
    normalized_laplacian: bool = True,
):
    """
    Analytic formula for the heat kernel on the Hamming graph, see
    Equation (3) in :cite:t:`doumont2025`.

    :param lengthscale:
        The length scale of the kernel, an array of shape [1].
    :param X:
        A batch of inputs, an array of shape [N, d].
    :param X2:
        A batch of inputs, an array of shape [N2, d]. If None, defaults to X.
    :param q:
        The alphabet size (number of categories). Defaults to 2 (hypercube).
    :param normalized_laplacian:
        Whether to use normalized Laplacian scaling.

    :return:
        The kernel matrix, an array of shape [N, N2].
    """
    return B.exp(
        _log_hamming_graph_heat_kernel(lengthscale, X, X2, q, normalized_laplacian)
    )


def _log_hamming_graph_heat_kernel(
    lengthscale, X, X2=None, q=2, normalized_laplacian=True
):
    """Log of the unit-diagonal heat kernel, with stable small-time evaluation."""
    if X2 is None:
        X2 = X
    _check_1_vector(lengthscale, "lengthscale")
    _check_matrix(X, "X")
    _check_matrix(X2, "X2")
    d = X.shape[-1]

    if normalized_laplacian:
        lengthscale = lengthscale / sqrt(d * (q - 1))

    beta = lengthscale**2 / 2

    # One for coordinates that differ, zero for coordinates that match.
    # Shape: [N, N2, d].
    disagreement = B.cast(B.dtype(lengthscale), X[:, None, :] != X2[None, :, :])

    exp_neg_beta_q = B.exp(-beta * q)
    # expm1 avoids cancellation in 1 - exp(-beta * q) at small lengthscales.
    factor_disagree = -expm1(-beta * q) / (1 + (q - 1) * exp_neg_beta_q)
    # Matching coordinates contribute log(1) = 0, including at zero lengthscale.
    # Mask before taking the logarithm to avoid 0 * log(0).
    safe_factor = B.where(disagreement == 0, B.ones(disagreement), factor_disagree)
    log_kernel = B.sum(B.log(safe_factor) * disagreement, axis=-1)

    return log_kernel
