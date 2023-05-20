"""
Space of Symmetric Positive-Definite Matrices.
"""

import geomstats as gs
import lab as B

from geometric_kernels.lab_extras import create_complex, dtype_double, slogdet, qr
from geometric_kernels.spaces import NoncompactSymmetricSpace
from geometric_kernels.utils.utils import ordered_pairwise_differences


class SymmetricPositiveDefiniteMatrices(NoncompactSymmetricSpace, gs.geometry.spd_matrices.SPDMatrices):
    r"""
    Manifold of symmetric positive-definite matrices.
    """

    def __init__(super, n):
        dim = n * (n+1) / 2
        super().__init__(dim=dim)
        self.n = n

    @property
    def dimension(self) -> int:
        return self.dim

    @property
    def degree(self) -> int:
        return self.n

    @property
    def rho(self):
        return (B.range(self.degree) + 1) / 2 - (self.degree + 1) / 4

    def random_phases(self, key, num):
        if not isinstance(num, tuple):
            num = (num, )
        x = B.randn(key, dtype_double(key), *num, self.degree, self.degree)
        Q, R = qr(x)
        r_diag_sign = B.sign(B.diag_extract(R))  # [B, N]
        Q *= B.expand_dims(r_diag_sign, -1)
        sign_det, _ = slogdet(Q)  # [B, ]
        Q[..., 0] *= B.expand_dims(sign_det, -1)
        return Q

    def inv_harish_chandra(self, X):
        """
        X shape [B, D]
        """
        diffX = ordered_pairwise_differences(X)
        diffX = B.pi * B.abs(diffX)
        logprod = B.sum(B.log(diffX) + B.log(B.tanh(diffX)), axis=-1)  # [B, ]
        return B.exp(0.5 * logprod)

    def power_function(self, lam, g, h):
        """
        :param lam: [N1, ..., Nk, D] eigenvalues.
        :param g: [N1, ..., Nk, D, D] matrices.
        :param h: [N1, ..., Nk, D, D] phases (orhogonal matrices).
        """

        gh = B.matmul(g, h)
        Q, R = qr(gh)
        u = B.diag_extract(R)  # [..., D]
        exponent = create_complex(self.rho, lam)  # [..., D]
        logpower = u * exponent  # [..., D]
        logproduct = B.sum(logpower, dim=-1)  # [...,]
        logproduct = B.cast(B.dtype(lam), logproduct)
        return B.exp(logproduct)
