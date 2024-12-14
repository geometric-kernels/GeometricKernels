"""
This module provides the :class:`Hyperbolic` space.
"""

import geomstats as gs
import lab as B
from beartype.typing import Optional

from geometric_kernels.lab_extras import complex_like, create_complex, dtype_double
from geometric_kernels.spaces.base import NoncompactSymmetricSpace
from geometric_kernels.utils.manifold_utils import (
    hyperbolic_distance,
    minkowski_inner_product,
)


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
        mathematical details in :cite:t:`azangulov2024b`.

    .. admonition:: Citation

        If you use this GeometricKernels space in your research, please consider
        citing :cite:t:`azangulov2024b`.
    """

    def __init__(self, dim=2):
        super().__init__(dim=dim)

    def __str__(self):
        return f"Hyperbolic({self.dimension})"

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
        Calls :func:`~.hyperbolic_distance` on the same inputs.
        """
        return hyperbolic_distance(x1, x2, diag)

    def inner_product(self, vector_a, vector_b):
        r"""
        Calls :func:`~.minkowski_inner_product` on `vector_a` and `vector_b`.
        """
        return minkowski_inner_product(vector_a, vector_b)

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
        Non-uniform random sampling, reimplements the algorithm from geomstats.

        Always returns [N, n+1] float64 array of the `key`'s backend.

        :param key:
            Either `np.random.RandomState`, `tf.random.Generator`,
            `torch.Generator` or `jax.tensor` (representing random state).
        :param number:
            Number of samples to draw.

        :return:
            An array of `number` random samples on the space.
        """

        key, samples = B.rand(key, dtype_double(key), number, self.dim)

        samples = 2.0 * (samples - 0.5)

        coord_0 = B.sqrt(1.0 + B.sum(samples**2, axis=-1))
        return key, B.concat(coord_0[..., None], samples, axis=-1)

    def element_shape(self):
        """
        :return:
            [n+1].
        """
        return [self.dimension + 1]

    @property
    def element_dtype(self):
        """
        :return:
            B.Float.
        """
        return B.Float
