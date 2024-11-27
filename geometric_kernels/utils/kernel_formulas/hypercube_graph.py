"""
Implements the closed form expression for the heat kernel on the hypercube graph.

The implementation is provided mainly for testing purposes.
"""

from math import sqrt

import lab as B
from beartype.typing import Optional

from geometric_kernels.lab_extras import float_like
from geometric_kernels.utils.utils import hamming_distance


def hypercube_graph_heat_kernel(
    lengthscale: B.Numeric,
    X: B.Numeric,
    X2: Optional[B.Numeric] = None,
    normalized_laplacian: bool = True,
):
    """
    Analytic formula for the heat kernel on the hypercube graph, see
    Equation (14) in :cite:t:`borovitskiy2023`.

    :param lengthscale:
        The length scale of the kernel, an array of shape [1].
    :param X:
        A batch of inputs, an array of shape [N, d].
    :param X2:
        A batch of inputs, an array of shape [N2, d].  If None, defaults to X.

    :return:
        The kernel matrix, an array of shape [N, N2].
    """
    if X2 is None:
        X2 = X

    assert lengthscale.shape == (1,)
    assert X.ndim == 2 and X2.ndim == 2
    assert X.shape[-1] == X2.shape[-1]

    if normalized_laplacian:
        d = X.shape[-1]
        lengthscale = lengthscale / sqrt(d)

    # For TensorFlow, we need to explicitly cast the distances to double.
    # Note: if we use B.dtype_float(X) instead of float_like(X), it gives
    # float16 and TensorFlow is still complaining.
    hamming_distances = B.cast(float_like(X), hamming_distance(X, X2))

    return B.tanh(lengthscale**2 / 2) ** hamming_distances
