"""
This module provides the :class:`SymmetricPositiveDefiniteMatrices` space.
"""

import geomstats as gs
import lab as B

from geometric_kernels.lab_extras import (
    create_complex,
    dtype_complex,
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
    definite matrices :math:`SPD(n)` with the affine-invariant Riemannian
    metric.

    The elements of this space are represented by positive definite matrices of
    size n x n.

    **Note:** positive definite means _strictly_ positive definite here, not
    positive semi-definite.

    The class inherits the interface of geomstats's `SPDMatrices`.
    """

    def __init__(self, n):
        super().__init__(n)

    @property
    def dimension(self) -> int:
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

    def inv_harish_chandra(self, X):
        """
        X shape [B, D]
        """
        diffX = ordered_pairwise_differences(X)
        diffX = B.abs(diffX)
        logprod = B.sum(
            B.log(B.pi * diffX) + B.log(B.tanh(B.pi * diffX)), axis=-1
        )  # [B, ]
        return B.exp(0.5 * logprod)

    def power_function(self, lam, g, h):
        """
        :param lam: [N1, ..., Nk, D] eigenvalues.
        :param g: [N1, ..., Nk, D, D] matrices.
        :param h: [N1, ..., Nk, D, D] phases (orhogonal matrices).
        """
        g = B.cholesky(g)
        gh = B.matmul(g, h)
        Q, R = qr(gh)

        u = B.abs(B.diag_extract(R))
        logu = B.cast(dtype_complex(R), B.log(u))
        exponent = create_complex(from_numpy(lam, self.rho), lam)  # [..., D]
        logpower = logu * exponent  # [..., D]
        logproduct = B.sum(logpower, axis=-1)  # [...,]
        logproduct = B.cast(dtype_complex(lam), logproduct)
        return B.exp(logproduct)

    def random(self, key, number):
        """
        Random points on the SPD space. Calls the respective routine of
        geomstats.

        TODO: implement in a way that actually uses key for randomness.

        Always returns [N, D, D] float64 array of the `key`'s backend.
        """

        return key, B.cast(dtype_double(key), self.random_point(number))
