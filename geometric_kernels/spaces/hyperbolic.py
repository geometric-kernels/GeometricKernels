"""
Hyperbolic space.
"""

from typing import Optional

import geomstats as gs
import lab as B
from opt_einsum import contract as einsum

from geometric_kernels.lab_extras import cosh, from_numpy, logspace, sinh, trapz, dtype_double
from geometric_kernels.spaces import NoncompactSymmetricSpace


class Hyperbolic(NoncompactSymmetricSpace, gs.geometry.hyperboloid.Hyperboloid):
    r"""
    Hyperbolic manifold.

    The class implements the hyperboloid model of the hyperbolic space :math:`H^n`.

    :math:`H^n = \{ (x_0, \ldots, x_{n}) | x_0^2 - \sum_{i=1}^{n} x_i^2 = 1, x_0 > 0 \}`

    The class inherits the interface of geomstats's `Hyperbolic` with `point_type=extrinsic`.
    """

    def __init__(self, dim=1):
        super().__init__(dim=dim)

    @property
    def dimension(self) -> int:
        return self.dim

    def distance(
        self, x1: B.Numeric, x2: B.Numeric, diag: Optional[bool] = False
    ) -> B.Numeric:
        """Compute the hyperbolic distance between `x1` and `x2`.

        The code is a reimplementation of `geomstats.geometry.hyperboloid.HyperbolicMetric` for `lab`.

        :param x1: [N, dim+1] array of points in the hyperbolic space
        :param x2: [M, dim+1] array of points in the hyperbolic space
        :param diag: if True, compute elementwise distance. Default False.
        :return: hyperbolic distance.
        """
        if diag:
            # Compute a pointwise distance between `x1` and `x2`
            x1_ = x1
            x2_ = x2
        else:
            if B.rank(x1) == 1:
                x1 = B.expand_dims(x1)
            if B.rank(x2) == 1:
                x2 = B.expand_dims(x2)

            # compute pairwise distance between arrays of points `x1` and `x2`
            # `x1` (N, dim+1)
            # `x2` (M, dim+1)
            x1_ = B.tile(x1[..., None, :], 1, x2.shape[0], 1)  # (N, M, dim+1)
            x2_ = B.tile(x2[None], x1.shape[0], 1, 1)  # (N, M, dim+1)

        sq_norm_1 = self.inner_product(x1_, x1_)
        sq_norm_2 = self.inner_product(x2_, x2_)
        inner_prod = self.inner_product(x1_, x2_)

        cosh_angle = -inner_prod / B.sqrt(sq_norm_1 * sq_norm_2)

        one = B.cast(B.dtype(cosh_angle), from_numpy(cosh_angle, [1.0]))
        large_constant = B.cast(B.dtype(cosh_angle), from_numpy(cosh_angle, [1e24]))

        # clip values into [1.0, 1e24]
        cosh_angle = B.where(cosh_angle < one, one, cosh_angle)
        cosh_angle = B.where(cosh_angle > large_constant, large_constant, cosh_angle)

        dist = B.log(cosh_angle + B.sqrt(cosh_angle**2 - 1))  # arccosh
        dist = B.cast(B.dtype(x1_), dist)
        return dist

    def inner_product(self, vector_a, vector_b):
        q = self.dimension
        p = 1
        diagonal = from_numpy(vector_a, [-1.0] * p + [1.0] * q)  # (dim+1)
        diagonal = B.cast(B.dtype(vector_a), diagonal)
        return einsum("...i,...i->...", diagonal * vector_a, vector_b)

    def inv_harish_chandra(self, X):
        if self.dimension % 2 == 0:
            m = self.dimension // 2
            js = (B.range(B.dtype(X), 2, m)*2 - 3.0)**2 / 4  # [M]
        elif self.dimension % 2 == 1:
            m = self.dimension // 2
            js = B.range(B.dtype(X), 0, m-1)**2  # [M]
        log_c = B.sum(2*B.log(X)+B.log(js), axis=1)  # [N, M] --> [N, ]
        if self.dimension % 2 == 0:
            log_c += B.log(X) + B.log(B.tanh(3.14 * X))

        return B.exp(0.5 * log_c)

    def power_function(self, lam, g, h):
        r"""
        Power function :math:`p^{\lambda)(g, h) = \exp(i \lambda + \rho) a(h \cdot g)`.

        Zonal spherical functions are defined as :math:`\pi^{\lambda}(g) = \int_{H} p^{\lambda}(g, h) d\mu_H(h).

        In the hyperbolic case, in Poincare ball coordinates,

        \exp(i \lambda + \rho) a(h \cdot g) = ((1-|g|^2)/|g-h|^2)^{-i\lambda+\rho}
        `
        """
        # lam [N1, .., Nk]
        # g [N1, ..., Nk, D]
        # h [N1, ..., Nk, D]
        # lam <-> lmd, g <-> x, h <-> shift
        g_poincare = self.convert_to_ball(g)
        gh_norm = B.sum(B.power(g_poincare-h, 2), axis=-1)  # [N1, ..., Nk]
        denominator = B.log(gh_norm)
        numerator = B.log(B.ones(gh_norm) - B.sum(g_poincare**2, axis=1))
        log_out = (numerator - denominator) * (-1j * lam + self.rho)  # [N1, ..., Nk]
        out = B.exp(log_out)
        return out

    def convert_to_ball(self, point):
        # point [N1, ..., Nk, D]
        return point[..., 1:] / (1 + point[..., :1])

    @property
    def rho(self):
        return (self.dimension - 1) / 2

    def random_phases(self, key, num):
        key, x = B.randn(key, dtype_double(key), num, self.dimension)
        x = x / B.sum(x**2, axis=-1, squeeze=False)
        return key, x

    def heat_kernel(
        self, distance: B.Numeric, t: B.Numeric, num_points: int = 100
    ) -> B.Numeric:
        """
        Compute the heat kernel associated with the space.

        We use Millson's formula for the heat kernel.

        References:
            [1] A. Grigoryan and M. Noguchi,
            The heat kernel on hyperbolic space.
            Bulletin of the London Mathematical Society, 30(6):643–650, 1998.

        :param distance: precomputed distance between the inputs
        :param t: heat kernel lengthscale
        :param num_points: number of points in the integral
        :return: heat kernel values
        """
        if self.dimension == 1:
            heat_kernel = B.exp(
                -B.power(distance[..., None], 2) / (4 * t)
            )  # (..., N1, N2, T)

        elif self.dimension == 2:
            expanded_distance = B.expand_dims(
                B.expand_dims(distance, -1), -1
            )  # (... N1, N2) -> (..., N1, N2, 1, 1)

            # TODO: the behavior of this kernel is not so stable around zero distance
            # due to the division in the computation of the integral value and
            # depends on the start of the s_vals interval
            s_vals = (
                B.cast(
                    B.dtype(expanded_distance),
                    from_numpy(
                        expanded_distance,
                        logspace(
                            B.log(1e-2), B.log(100.0), num_points, base=B.exp(1.0)
                        ),
                    ),
                )
                + expanded_distance
            )  # (..., N1, N2, 1, S)
            reshape = [1] * B.rank(s_vals)
            reshape[-2] = B.shape(t)[-1]
            s_vals = B.tile(s_vals, *reshape)  # (N1, N2, T, S)
            integral_vals = (
                s_vals
                * B.exp(-(s_vals**2) / (4 * t[:, None]))
                / B.sqrt(cosh(s_vals) - cosh(expanded_distance))
            )  # (..., N1, N2, T, S)

            integral_vals = B.cast(B.dtype(s_vals), integral_vals)

            heat_kernel = trapz(integral_vals, s_vals, axis=-1)  # (..., N1, N2, T)

        elif self.dimension == 3:
            heat_kernel = B.exp(
                -B.power(distance[..., None], 2) / (4 * t)
            )  # (..., N1, N2, T)
            heat_kernel = (
                heat_kernel
                * (distance[..., None] + 1e-8)
                / sinh(distance[..., None] + 1e-8)
            )  # Adding 1e-8 avoids numerical issues around d=0

        else:
            raise NotImplementedError

        return heat_kernel
