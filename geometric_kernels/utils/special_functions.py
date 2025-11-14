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
from geometric_kernels.utils.utils import _check_matrix


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
    _check_matrix(x, "x")
    if B.shape(x)[-1] != d:
        raise ValueError("`x` must live in `d`-dimensional space.")

    indices = B.cast(int_like(x), from_numpy(x, combination))[None, :]

    return (-1) ** count_nonzero(take_along_axis(x, indices, axis=-1), axis=-1)


def generalized_kravchuk_normalized(
    d: int,
    j: int,
    m: B.Int,
    q: int,
    kravchuk_normalized_j_minus_1: Optional[B.Float] = None,
    kravchuk_normalized_j_minus_2: Optional[B.Float] = None,
) -> B.Float:
    r"""
    This function returns $\widetilde{G}_{d, q, j}(m)$ where $\widetilde{G}_{d, q, j}(m)$ is the
    normalized generalized Kravchuk polynomial defined below.

    Define the generalized Kravchuk polynomial of degree $d > 0$ and order $0 \leq j \leq d$
    as the function $G_{d, q, j}(m)$ of the independent variable $0 \leq m \leq d$ given by

    .. math:: G_{d, q, j}(m) = \sum_{\substack{T \subseteq \{0, .., d-1\} \\ |T| = j}}
              \sum_{\alpha \in \{1,..,q-1\}^T} \psi_{T,\alpha}(x) \overline{\psi_{T,\alpha}(y)}.

    Here $\psi_{T,\alpha}$ are the Vilenkin functions on the q-ary Hamming graph $H(d,q)$ and
    $x, y \in \{0, 1, ..., q-1\}^d$ are arbitrary categorical vectors at Hamming distance $m$
    (the right-hand side does not depend on the choice of particular vectors of the kind).

    The normalized polynomial is $\widetilde{G}_{d, q, j}(m) = G_{d, q, j}(m) / G_{d, q, j}(0)$.

    .. note::
        We are using the three term recurrence relation to compute the Kravchuk
        polynomials. Cf. Equation (60) of Chapter 5 in MacWilliams and Sloane "The
        Theory of Error-Correcting Codes", 1977. The parameter $\gamma$ from
        :cite:t:`macwilliams1977` is set to $\gamma = q - 1$.

    .. note::
        We use the fact that $G_{d, q, j}(0) = \binom{d}{j}(q-1)^j$.

    .. note::
        For $q = 2$, this reduces to the classical Kravchuk polynomials.

    :param d:
        The degree of Kravchuk polynomial, an integer $d > 0$.
        Maps to $n$ in :cite:t:`macwilliams1977`.

    :param j:
        The order of Kravchuk polynomial, an integer $0 \leq j \leq d$.
        Maps to $k$ in :cite:t:`macwilliams1977`.

    :param m:
        The independent variable (Hamming distance), an integer or array with
        $0 \leq m \leq d$. Maps to $x$ in :cite:t:`macwilliams1977`.

    :param q:
        The alphabet size of the q-ary Hamming graph, an integer $q \geq 2$.

    :param kravchuk_normalized_j_minus_1:
        The optional precomputed value of $\widetilde{G}_{d, q, j-1}(m)$, helps
        to avoid exponential complexity growth due to the recursion.

    :param kravchuk_normalized_j_minus_2:
        The optional precomputed value of $\widetilde{G}_{d, q, j-2}(m)$, helps
        to avoid exponential complexity growth due to the recursion.

    :return:
        $\widetilde{G}_{d, q, j}(m) = G_{d, q, j}(m) / G_{d, q, j}(0)$ where
        $G_{d, q, j}(m)$ is the generalized Kravchuk polynomial.
    """
    if d <= 0:
        raise ValueError("`d` must be positive.")
    if not (0 <= j and j <= d):
        raise ValueError("`j` must lie in the interval [0, d].")
    if not (B.all(0 <= m) and B.all(m <= d)):
        raise ValueError("`m` must lie in the interval [0, d].")

    m = B.cast(B.dtype_float(m), m)

    if j == 0:
        return B.ones(m)
    elif j == 1:
        return 1 - q * m / (d * (q - 1))
    else:
        if kravchuk_normalized_j_minus_1 is None:
            kravchuk_normalized_j_minus_1 = generalized_kravchuk_normalized(
                d, j - 1, m, q
            )
        if kravchuk_normalized_j_minus_2 is None:
            kravchuk_normalized_j_minus_2 = generalized_kravchuk_normalized(
                d, j - 2, m, q
            )

        rhs_1 = (
            (d - j + 1) * (q - 1) + (j - 1) - q * m
        ) * kravchuk_normalized_j_minus_1
        rhs_2 = -(j - 1) * kravchuk_normalized_j_minus_2

        return (rhs_1 + rhs_2) / ((d - j + 1) * (q - 1))
