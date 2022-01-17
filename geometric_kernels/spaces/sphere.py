"""
Hypersphere space.
"""

from typing import Optional

import math
import geomstats as gs
import lab as B

from geometric_kernels.spaces import Space


def gegenbauer_polynomial(n, alpha, z):
    """
    This function computes the Gegenbauer polynomial C_n^alpha(z).

    Parameters
    ----------
    :param n: Gegenbauer polynomial parameter n
    :param alpha: Gegenbauer polynomial parameter alpha
    :param z: Gegenbauer polynomial function input

    Returns
    -------
    :return: Gegenbauer polynomial C_n^alpha(z)

    """
    # Initialization
    polynomial = 0.
    gamma_alpha = math.gamma(alpha)

    # Computes the summation serie
    for i in range(math.floor(n / 2) + 1):
        polynomial += math.pow(-1, i) * B.power(2 * z, n - 2 * i) \
                      * (math.gamma(n - i + alpha) / (gamma_alpha * math.factorial(i) * math.factorial(n - 2 * i)))

    return polynomial


def sphere_kernel_constant(n, d):
    """
    This function computes the constant terms of the summation formula for the Matérn kernel on the sphere.

    Parameters
    ----------
    :param n: term n of the summation serie
    :param d: dimension of sphere S^d

    Returns
    -------
    :return: constant term c_nd

    """
    dn = (2*n+d-1) * math.gamma(n+d-1) / math.gamma(d) / math.gamma(n+1)
    # Compute Gegenbauer polynomial
    gpolynomial = gegenbauer_polynomial(n, (d+1.)/2., B.ones(1))
    # Constant value
    cnd = dn * math.gamma((d+1.)/2.) / (2*math.pow(math.pi, (d+1.)/2.) * gpolynomial)
    return cnd


class Sphere(Space, gs.geometry.hypersphere.Hypersphere):
    def __init__(self, dim=2):
        super().__init__(dim=dim)

    @property
    def dimension(self) -> int:
        return self.dim

    @property
    def is_compact(self) -> bool:
        return True

    def distance(
            self, x1: B.Numeric, x2: B.Numeric, diag: Optional[bool] = False
    ) -> B.Numeric:
        assert B.all(self.belongs(x1)) and B.all(self.belongs(x2))

        return self.metric.dist(x1, x2)

    def heat_kernel(
            self, distance: B.Numeric, t: B.Numeric, num_terms: int = 10
    ) -> B.Numeric:
        """
        Compute the heat kernel associated with the space.

        We use the serie formulation based on Gegenbauer polynomials.

        References:
        [1] V. Borovitskiy, A. Terenin, P. Mostowsky, and M. P. Deisenroth.
        Matérn Gaussian processes on Riemannian manifolds.
        In: NeurIPS 2020.

        Parameters
        ----------
        :param distance: precomputed distance between the inputs
        :param t: heat kernel lengthscale
        :param num_terms: number of terms in the serie

        Returns
        -------
        :return: heat kernel values
        """
        if isinstance(distance, float):
            distance = B.ones(1) * distance

        # Compute cos of distance
        cos_distance = B.cos(distance)

        # Compute constant terms
        cst_nd = [sphere_kernel_constant(n, self.dim) for n in range(num_terms)]

        # Compute serie
        heat_kernel = B.zeros(distance.dtype, *distance.shape, *t.shape)
        # heat_kernel = B.zeros(distance.dtype, *distance.shape, *t.shape)
        for n in range(num_terms):
            # Compute exponential term
            exp_term = B.exp(- t * n * (n + self.dim - 1))
            # Compute Gegenbauer polynomial
            gpolynomial = gegenbauer_polynomial(n, (self.dim - 1.) / 2., cos_distance)
            # Kernel serie's n-th term
            for i in range(t.shape[0]):
                heat_kernel[..., i] += exp_term[i] * cst_nd[n] * gpolynomial

        # Kernel
        return heat_kernel
