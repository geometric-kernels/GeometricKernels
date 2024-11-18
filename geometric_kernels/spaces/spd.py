"""
This module provides the :class:`SymmetricPositiveDefiniteMatrices` space.
"""

import geomstats as gs
import lab as B

from geometric_kernels.lab_extras import (
    complex_like,
    create_complex,
    dtype_double,
    from_numpy,
    qr,
    slogdet,
)
from geometric_kernels.spaces.base import NoncompactSymmetricSpace
from geometric_kernels.utils.utils import ordered_pairwise_differences


class SymmetricPositiveDefiniteMatrices(
    NoncompactSymmetricSpace, gs.geometry.spd_matrices.SPDMatrices
):
    r"""
    The GeometricKernels space representing the manifold of symmetric positive
    definite matrices $SPD(n)$ with the affine-invariant Riemannian metric.

    The elements of this space are represented by positive definite matrices of
    size n x n. Positive definite means _strictly_ positive definite here, not
    positive semi-definite.

    The class inherits the interface of geomstats's `SPDMatrices`.

    .. note::
        A tutorial on how to use this space is available in the
        :doc:`SPD.ipynb </examples/SPD>` notebook.

    :param n:
        Size of the matrices, the $n$ in $SPD(n)$.

    .. note::
        As mentioned in :ref:`this note <quotient note>`, any symmetric space
        is a quotient G/H. For the manifold of symmetric positive definite
        matrices $SPD(n)$, the group of symmetries $G$ is the identity component
        $GL(n)_+$ of the general linear group $GL(n)$, while the isotropy
        subgroup $H$ is the special orthogonal group $SO(n)$. See the
        mathematical details in :cite:t:`azangulov2024b`.

    .. admonition:: Citation

        If you use this GeometricKernels space in your research, please consider
        citing :cite:t:`azangulov2024b`.
    """

    def __init__(self, n):
        super().__init__(n)

    def __str__(self):
        return f"SymmetricPositiveDefiniteMatrices({self.n})"

    @property
    def dimension(self) -> int:
        """
        Returns n(n+1)/2 where `n` was passed down to `__init__`.
        """
        dim = self.n * (self.n + 1) / 2
        return dim

    @property
    def degree(self) -> int:
        return self.n

    @property
    def rho(self):
        return (B.range(self.degree) + 1) - (self.degree + 1) / 2

    @property
    def num_axes(self):
        """
        Number of axes in an array representing a point in the space.

        :return:
            2.
        """
        return 2

    def random_phases(self, key, num):
        if not isinstance(num, tuple):
            num = (num,)
        key, x = B.randn(key, dtype_double(key), *num, self.degree, self.degree)
        Q, R = qr(x)
        r_diag_sign = B.sign(B.diag_extract(R))  # [B, N]
        Q *= B.expand_dims(r_diag_sign, -1)  # [B, D, D]
        sign_det, _ = slogdet(Q)  # [B, ]

        # equivalent to Q[..., 0] *= B.expand_dims(sign_det, -1)
        Q0 = Q[..., 0] * B.expand_dims(sign_det, -1)  # [B, D]
        Q = B.concat(B.expand_dims(Q0, -1), Q[..., 1:], axis=-1)  # [B, D, D]
        return key, Q

    def inv_harish_chandra(self, lam):
        diffs = ordered_pairwise_differences(lam)
        diffs = B.abs(diffs)
        logprod = B.sum(
            B.log(B.pi * diffs) + B.log(B.tanh(B.pi * diffs)), axis=-1
        )  # [B, ]
        return B.exp(0.5 * logprod)

    def power_function(self, lam, g, h):
        g = B.cholesky(g)
        gh = B.matmul(g, h)
        Q, R = qr(gh)

        u = B.abs(B.diag_extract(R))
        logu = B.cast(complex_like(R), B.log(u))
        exponent = create_complex(from_numpy(lam, self.rho), lam)  # [..., D]
        logpower = logu * exponent  # [..., D]
        logproduct = B.sum(logpower, axis=-1)  # [...,]
        logproduct = B.cast(complex_like(lam), logproduct)
        return B.exp(logproduct)

    def random(self, key, number):
        """
        Non-uniform random sampling, reimplements the algorithm from geomstats.

        Always returns [N, n, n] float64 array of the `key`'s backend.

        :param key:
            Either `np.random.RandomState`, `tf.random.Generator`,
            `torch.Generator` or `jax.tensor` (representing random state).
        :param number:
            Number of samples to draw.

        :return:
            An array of `number` random samples on the space.
        """

        key, mat = B.rand(key, dtype_double(key), number, self.n, self.n)
        mat = 2 * mat - 1
        mat_symm = 0.5 * (mat + B.transpose(mat, (0, 2, 1)))

        return key, B.expm(mat_symm)

    @property
    def element_shape(self):
        """
        :return:
            [n, n].
        """
        return [self.n, self.n]

    @property
    def element_dtype(self):
        """
        :return:
            B.Float.
        """
        return B.Float
