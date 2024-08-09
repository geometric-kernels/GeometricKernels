"""
Special mathematical functions used in the library.
"""

from math import sqrt

import lab as B
from beartype.typing import List, Optional

from geometric_kernels.lab_extras import (
    count_nonzero,
    float_like,
    from_numpy,
    int_like,
    take_along_axis,
)
from geometric_kernels.utils.utils import hamming_distance


def walsh_function(d: int, combination: List[int], x: B.Bool) -> B.Float:
    r"""
    This function returns the value of the Walsh function

    .. math:: w_T(x_0, .., x_{d-1}) = (-1)^{\sum_{j \in T} x_j}

    where $d$ is `d`, $T$ is `combination`, and $x = (x_0, .., x_{d-1})$ is `x`.

    :param d:
        The degree of the Walsh function and the dimension of its inputs.
        An integer d > 0.
    :param combination:
        A subset of the set $\{0, .., d-1\}$ determining the particular Walsh
        function. A list of integers.
    :param x:
        A batch of binary vectors $x = (x_0, .., x_{d-1})$ of shape [N, d].

    :return:
        The value of the Walsh function $w_T(x)$ evaluated for every $x$ in the
        batch. An array of shape [N].

    """
    assert x.ndim == 2
    assert x.shape[-1] == d

    indices = B.cast(int_like(x), from_numpy(x, combination))[None, :]

    return (-1) ** count_nonzero(take_along_axis(x, indices, axis=-1), axis=-1)


def kravchuk_normalized(d: int, j: int, m: B.Int) -> B.Float:
    r"""
    This function returns $G_{d, j, m}/G_{d, j, 0}$ where $G_{d, j, m}$ is the
    Kravchuk polynomial defined below.

    Define the Kravchuk polynomial of degree d > 0 and order 0 <= j <= d as the
    function $G_{d, j, m}$ of the independent variable 0 <= m <= d given by

    .. math:: G_{d, j, m} = \sum_{T \subseteq \{0, .., d-1\}, |T| = j} w_T(x).

    Here $w_T$ are the Walsh functions on the hypercube $C^d = \{0, 1\}^d$ and
    $x \in C^d$ is an arbitrary binary vector with $m$ ones (the right-hand side
    does not depend on the choice of a particular vector of the kind).

    .. note::
        We are using the three term recurrence relation to compute the Kravchuk
        polynomials. Cf. Equation (60) in MacWilliams and Sloane "The Theory of
        Error-Correcting Codes", 1977. The parameters q and \gamma from
        :cite:t:`macwilliams1977` are set to be q = 2; \gamma = q - 1 = 1.

    .. note::
        We use the fact that $G_{d, j, 0} = \binom{d}{j}$.

    :param d:
        The degree of Kravhuk polynomial, an integer d > 0.
        Maps to n in :cite:t:`macwilliams1977`.
    :param j: d
        The order of Kravhuk polynomial, an integer 0 <= j <= d.
        Maps to k in :cite:t:`macwilliams1977`.
    :param m:
        The independent variable, an integer 0 <= m <= d.
        Maps to x in :cite:t:`macwilliams1977`.

    :return:
        $G_{d, j, m}/G_{d, j, 0}$ where $G_{d, j, m}$ is the Kravchuk polynomial.
    """
    assert d > 0
    assert 0 <= j and j <= d
    assert B.all(0 <= m) and B.all(m <= d)

    m = B.cast(B.dtype_float(m), m)

    if j == 0:
        return 1 + 0 * m  # 0*m is a hack to make the output have the same shape as m
    elif j == 1:
        return 1 - 2 * m / d
    else:
        rhs_1 = (d - 2 * m) * kravchuk_normalized(d, j - 1, m)
        rhs_2 = -(j - 1) * kravchuk_normalized(d, j - 2, m)
        return (rhs_1 + rhs_2) / (d - j + 1)


def hypercube_heat_kernel(
    lengthscale: B.Numeric,
    X: B.Numeric,
    X2: Optional[B.Numeric] = None,
    normalized_laplacian: bool = True,
):
    """
    Analytic formula for the heat kernel on the hypercube, see Equation (14) in
    :cite:t:`borovitskiy2023`.

    :param lengthscale:
        The length scale of the kernel, an array of shape [1].
    :param X:
        A batch of inputs, an array of shape [N, d].
    :param X2:
        A batch of inputs, an array of shape [N2, d].

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
