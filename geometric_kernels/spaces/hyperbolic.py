"""
This module provides the :class:`Hyperbolic` space.
"""

import geomstats as gs
import lab as B
from beartype.typing import Optional
from opt_einsum import contract as einsum

from geometric_kernels.lab_extras import (
    complex_like,
    create_complex,
    dtype_double,
    from_numpy,
)
from geometric_kernels.spaces.base import NoncompactSymmetricSpace


class Hyperbolic(NoncompactSymmetricSpace, gs.geometry.hyperboloid.Hyperboloid):
    r"""
    The GeometricKernels space representing the Hyperbolic spaces :math:`H^n`.
    More specifically, we use the hyperboloid model of the hyperbolic space.

    The elements of this space are represented by (n+1)-dimensional vectors satisfying

    :math:`x_0^2 - x_1^2 - \ldots - x_{d}^2 = 1,`

    i.e. lying on the hyperboloid.

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
        X = B.squeeze(X, -1)
        if self.dimension == 2:
            c = B.abs(X) * B.tanh(B.pi * B.abs(X))
            return B.sqrt(c)

        if self.dimension % 2 == 0:
            m = self.dimension // 2
            js = B.range(B.dtype(X), 0, m - 1)
            addenda = ((js * 2 + 1.0) ** 2) / 4  # [M]
        elif self.dimension % 2 == 1:
            m = self.dimension // 2
            js = B.range(B.dtype(X), 0, m)
            addenda = js**2  # [M]
        log_c = B.sum(B.log(X[..., None] ** 2 + addenda), axis=-1)  # [N, M] --> [N, ]
        if self.dimension % 2 == 0:
            log_c += B.log(B.abs(X)) + B.log(B.tanh(B.pi * B.abs(X)))

        return B.exp(0.5 * log_c)

    def power_function(self, lam, g, h):
        r"""
        Power function :math:`p^{\lambda)(g, h) = \exp(i \lambda + \rho) a(h \cdot g)`.

        Zonal spherical functions are defined as :math:`\pi^{\lambda}(g) = \int_{H} p^{\lambda}(g, h) d\mu_H(h)`.

        In the hyperbolic case, in Poincare ball coordinates,
        :math:`\exp(i \lambda + \rho) a(h \cdot g) = ((1 - |g|^2)/|g - h|^2)^{-i |\lambda| + \rho}`

        :param lam: [N1, ..., Nk, 1] eigenvalues.
        :param g: [N1, ..., Nk, D+1] points on the hyperbolic space.
        :param h: [N1, ..., Nk, D] phases (points on the unit sphere).
        """
        lam = B.squeeze(lam, -1)
        g_poincare = self.convert_to_ball(g)  # [..., D]
        gh_norm = B.sum(B.power(g_poincare - h, 2), axis=-1)  # [N1, ..., Nk]
        denominator = B.log(gh_norm)
        numerator = B.log(1.0 - B.sum(g_poincare**2, axis=-1))
        exponent = create_complex(self.rho[0], -1 * B.abs(lam))  # rho is 1-d
        log_out = (
            B.cast(complex_like(lam), (numerator - denominator)) * exponent
        )  # [N1, ..., Nk]
        out = B.exp(log_out)
        return out

    def convert_to_ball(self, point):
        """
        Converts `point` from extrinsic coordinates (i.e, in the ambient :math:`R^{d+1}` space)
        to Poincare ball coordinates. This corresponds to stereographically projecting the hyperboloid
        onto the ball.
        """
        # point [N1, ..., Nk, D]
        return point[..., 1:] / (1 + point[..., :1])

    @property
    def rho(self):
        return B.ones(1) * (self.dimension - 1) / 2

    @property
    def num_axes(self):
        return 1

    def random_phases(self, key, num):
        if not isinstance(num, tuple):
            num = (num,)
        key, x = B.randn(key, dtype_double(key), *num, self.dimension)
        x = x / B.sqrt(B.sum(x**2, axis=-1, squeeze=False))
        return key, x

    def random(self, key, number):
        """
        Random points on the hyperbolic space. Calls the respective routine
        of geomstats.

        TODO: implement in a way that actually uses key for randomness.

        Always returns [N, D+1] float64 array of the `key`'s backend.
        """

        return key, B.cast(dtype_double(key), self.random_point(number))
