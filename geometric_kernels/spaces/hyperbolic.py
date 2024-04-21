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
    The GeometricKernels space representing the n-dimensional hyperbolic space
    $\mathbb{H}_n$. We use the hyperboloid model of the hyperbolic space.

    The elements of this space are represented by (n+1)-dimensional vectors
    satisfying

    .. math:: x_0^2 - x_1^2 - \ldots - x_n^2 = 1,

    i.e. lying on the hyperboloid.

    The class inherits the interface of geomstats's `Hyperbolic` with
    `point_type=extrinsic`.

    .. note::
        A tutorial on how to use this space is available in the
        :doc:`Hyperbolic.ipynb </examples/Hyperbolic>` notebook.

    :param dim:
        Dimension of the hyperbolic space, denoted by n in docstrings.

    .. note::
        As mentioned in :ref:`this note <quotient note>`, any symmetric space
        is a quotient G/H. For the hyperbolic space $\mathbb{H}_n$, the group
        of symmetries $G$ is the proper Lorentz group $SO(1, n)$,  while the
        isotropy subgroup $H$ is the special orthogonal group $SO(n)$. See the
        mathematical details in :cite:t:`azangulov2023`.

    .. admonition:: Citation

        If you use this GeometricKernels space in your research, please consider
        citing :cite:t:`azangulov2023`.
    """

    def __init__(self, dim=2):
        super().__init__(dim=dim)

    @property
    def dimension(self) -> int:
        """
        Returns n, the `dim` parameter that was passed down to `__init__`.
        """
        return self.dim

    def distance(
        self, x1: B.Numeric, x2: B.Numeric, diag: Optional[bool] = False
    ) -> B.Numeric:
        """
        Compute the hyperbolic distance between `x1` and `x2`.

        The code is a reimplementation of
        `geomstats.geometry.hyperboloid.HyperbolicMetric` for `lab`.

        :param x1:
            An [N, n+1]-shaped array of points in the hyperbolic space.
        :param x2:
            An [M, n+1]-shaped array of points in the hyperbolic space.
        :param diag:
            If True, compute elementwise distance. Requires N = M.

            Default False.

        :return:
            An [N, M]-shaped array if diag=False or [N,]-shaped array
            if diag=True.
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
            # `x1` (N, n+1)
            # `x2` (M, n+1)
            x1_ = B.tile(x1[..., None, :], 1, x2.shape[0], 1)  # (N, M, n+1)
            x2_ = B.tile(x2[None], x1.shape[0], 1, 1)  # (N, M, n+1)

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
        r"""
        Computes the Minkowski inner product of vectors.

        .. math:: \langle a, b \rangle = a_0 b_0 - a_1 b_1 - \ldots - a_n b_n.

        :param vector_a:
            An [..., n+1]-shaped array of points in the hyperbolic space.
        :param vector_b:
            An [..., n+1]-shaped array of points in the hyperbolic space.

        :return:
            An [...,]-shaped array of inner products.
        """
        q = self.dimension
        p = 1
        diagonal = from_numpy(vector_a, [-1.0] * p + [1.0] * q)  # (n+1)
        diagonal = B.cast(B.dtype(vector_a), diagonal)
        return einsum("...i,...i->...", diagonal * vector_a, vector_b)

    def inv_harish_chandra(self, lam: B.Numeric) -> B.Numeric:
        lam = B.squeeze(lam, -1)
        if self.dimension == 2:
            c = B.abs(lam) * B.tanh(B.pi * B.abs(lam))
            return B.sqrt(c)

        if self.dimension % 2 == 0:
            m = self.dimension // 2
            js = B.range(B.dtype(lam), 0, m - 1)
            addenda = ((js * 2 + 1.0) ** 2) / 4  # [M]
        elif self.dimension % 2 == 1:
            m = self.dimension // 2
            js = B.range(B.dtype(lam), 0, m)
            addenda = js**2  # [M]
        log_c = B.sum(B.log(lam[..., None] ** 2 + addenda), axis=-1)  # [N, M] --> [N, ]
        if self.dimension % 2 == 0:
            log_c += B.log(B.abs(lam)) + B.log(B.tanh(B.pi * B.abs(lam)))

        return B.exp(0.5 * log_c)

    def power_function(self, lam: B.Numeric, g: B.Numeric, h: B.Numeric) -> B.Numeric:
        lam = B.squeeze(lam, -1)
        g_poincare = self.convert_to_ball(g)  # [..., n]
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
        Converts the `point` from the hyperboloid model to the Poincare model.
        This corresponds to a stereographic projection onto the ball.

        :param:
            An [..., n+1]-shaped array of points on the hyperboloid.

        :return:
            An [..., n]-shaped array of points in the Poincare ball.
        """
        # point [N1, ..., Nk, n]
        return point[..., 1:] / (1 + point[..., :1])

    @property
    def rho(self):
        return B.ones(1) * (self.dimension - 1) / 2

    @property
    def num_axes(self):
        """
        Number of axes in an array representing a point in the space.

        :return:
            1.
        """
        return 1

    def random_phases(self, key, num):
        if not isinstance(num, tuple):
            num = (num,)
        key, x = B.randn(key, dtype_double(key), *num, self.dimension)
        x = x / B.sqrt(B.sum(x**2, axis=-1, squeeze=False))
        return key, x

    def random(self, key, number):
        """
        Geomstats-based non-uniform random sampling.

        Always returns [N, n+1] float64 array of the `key`'s backend.

        :param key:
            Either `np.random.RandomState`, `tf.random.Generator`,
            `torch.Generator` or `jax.tensor` (representing random state).
        :param number:
            Number of samples to draw.

        :return:
            An array of `number` uniformly random samples on the space.
        """

        return key, B.cast(dtype_double(key), self.random_point(number))

    def element_shape(self):
        """
        :return:
            [n+1].
        """
        return [self.dimension + 1]
