import lab as B
from beartype.typing import List

from geometric_kernels.lab_extras import from_numpy, int_like, take_along_axis


def walsh_function(d: int, combination: List[int], x: B.Int) -> B.Float:
    indices = B.cast(int_like(x), from_numpy(x, combination))[None, :]

    return (-1) ** B.sum(take_along_axis(x, indices, axis=-1), axis=-1)


def kravchuk_normalized(d: int, j: int, m: B.Int) -> B.Float:
    r"""
    This function returns $G_{d, j, m}/G_{d, j, 0}$ where $G_{d, j, m}$ is the
    Kravchuk polynomial defined below.

    Define the Kravchuk polynomial of degree d > 0 and order 0 <= j <= d as the
    function $G_{d, j, m}$ of the independent variable 0 <= m <= d given by

    .. math:: G_{d, j, m} = \sum_{T \subseteq \{1, .., d\}, |T| = j} w_T(x).

    Here $w_T$ are the Walsh functions on the hypercube $C = \{0, 1\}^d$ and
    $x \in C$ is an arbitrary binary vector with $m$ ones (the right-hand side
    does not depend on the choice of such vector).

    .. note::
        We are using the three term recurrence relation to compute the Kravchuk
        polynomials. Cf. Equation (60) in MacWilliams and Sloane "The Theory of
        Error-Correcting Codes", 1977. The parameters q and \gamma from
        MacWilliams and Sloane are set to be q = 2; \gamma = q - 1 = 1.

    .. note::
        We use the fact that $G_{d, j, 0} = binom{d}{j}$.

    :param d:
        The degree of Kravhuk polynomial, an integer d > 0.
        Maps to n in MacWilliams and Sloane.
    :param j: d
        The order of Kravhuk polynomial, an integer 0 <= j <= d.
        Maps to k in MacWilliams and Sloane.
    :param m:
        The independent variable, an integer 0 <= m <= d.
        Maps to x in MacWilliams and Sloane.

    :return:
        $G_{d, j, m}/G_{d, j, 0}$ where $G_{d, j, m}$ is the Kravchuk polynomial.
    """
    assert d > 0
    assert 0 <= j and j <= d
    assert B.all(0 <= m) and B.all(m <= d)

    if j == 0:
        return 1 + 0 * m  # 0*m is a hack to make the output have the same shape as m
    elif j == 1:
        return 1 - 2 * m / d
    else:
        rhs_1 = (d - 2 * m) * kravchuk_normalized(d, j - 1, m)
        rhs_2 = -(j - 1) * kravchuk_normalized(d, j - 2, m)
        return (rhs_1 + rhs_2) / (d - j + 1)
