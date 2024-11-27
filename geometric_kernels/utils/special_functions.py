"""
Special mathematical functions used in the library.
"""

import lab as B
from beartype.typing import List, Optional

from geometric_kernels.lab_extras import (
    count_nonzero,
    from_numpy,
    int_like,
    take_along_axis,
)


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


def kravchuk_normalized(
    d: int,
    j: int,
    m: B.Int,
    kravchuk_normalized_j_minus_1: Optional[B.Float] = None,
    kravchuk_normalized_j_minus_2: Optional[B.Float] = None,
) -> B.Float:
    r"""
    This function returns $G_{d, j, m}/G_{d, j, 0}$ where $G_{d, j, m}$ is the
    Kravchuk polynomial defined below.

    Define the Kravchuk polynomial of degree d > 0 and order 0 <= j <= d as the
    function $G_{d, j, m}$ of the independent variable 0 <= m <= d given by

    .. math:: G_{d, j, m} = \sum_{T \subseteq \{0, .., d-1\}, |T| = j} w_T(x).

    Here $w_T$ are the Walsh functions on the hypercube graph $C^d$ and
    $x \in C^d$ is an arbitrary binary vector with $m$ ones (the right-hand side
    does not depend on the choice of a particular vector of the kind).

    .. note::
        We are using the three term recurrence relation to compute the Kravchuk
        polynomials. Cf. Equation (60) of Chapter 5 in MacWilliams and Sloane "The
        Theory of Error-Correcting Codes", 1977. The parameters q and $\gamma$
        from :cite:t:`macwilliams1977` are set to be q = 2; $\gamma = q - 1 = 1$.

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
    :param kravchuk_normalized_j_minus_1:
        The optional precomputed value of $G_{d, j-1, m}/G_{d, j-1, 0}$, helps
        to avoid exponential complexity growth due to the recursion.
    :param kravchuk_normalized_j_minus_2:
        The optional precomputed value of $G_{d, j-2, m}/G_{d, j-2, 0}$, helps
        to avoid exponential complexity growth due to the recursion.

    :return:
        $G_{d, j, m}/G_{d, j, 0}$ where $G_{d, j, m}$ is the Kravchuk polynomial.
    """
    assert d > 0
    assert 0 <= j and j <= d
    assert B.all(0 <= m) and B.all(m <= d)

    m = B.cast(B.dtype_float(m), m)

    if j == 0:
        return B.ones(m)
    elif j == 1:
        return 1 - 2 * m / d
    else:
        if kravchuk_normalized_j_minus_1 is None:
            kravchuk_normalized_j_minus_1 = kravchuk_normalized(d, j - 1, m)
        if kravchuk_normalized_j_minus_2 is None:
            kravchuk_normalized_j_minus_2 = kravchuk_normalized(d, j - 2, m)
        rhs_1 = (d - 2 * m) * kravchuk_normalized_j_minus_1
        rhs_2 = -(j - 1) * kravchuk_normalized_j_minus_2
        return (rhs_1 + rhs_2) / (d - j + 1)
