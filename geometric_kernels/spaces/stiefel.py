import lab as B
import numpy as np
from opt_einsum import contract as einsum

from geometric_kernels.lab_extras import qr
from geometric_kernels.spaces.homogeneous_spaces import (
    AveragingAdditionTheorem,
    CompactHomogeneousSpace,
)
from geometric_kernels.spaces.so import SOGroup


def hook_content_formula(lmd, n):
    """
    A combinatorial formula used to calculate the dimension of invariant space of the Stiefeld manifold (among other things).
    """
    numer = 1
    denom = 1

    l_cols = [sum([row_l >= i + 1 for row_l in lmd]) for i in range(lmd[0])]
    for id_row, l_row in enumerate(lmd):
        for id_col in range(l_row):
            numer *= n + id_col - id_row
            denom *= l_cols[id_col] + l_row - id_row - id_col - 1

    return numer / denom


class StiefelEigenfunctions(AveragingAdditionTheorem):
    def _compute_projected_character_value_at_e(self, signature):
        """
        Value of character on class of identity element is equal to the dimension of invariant space.
        In case of Stiefel manifold it could be computed using the hook-content formula.
        :param signature: the character signature
        :return: int
        """

        m_ = min(self.M.m, self.M.n - self.M.m)
        if m_ < self.M.G.rank and signature[m_] > 0:
            return 0
        signature_abs = tuple(abs(x) for x in signature)
        return hook_content_formula(signature_abs, m_)


class Stiefel(CompactHomogeneousSpace):
    r"""
    Stifiel manifold `V(n, m) = SO(n) / SO(n-m)`.

    `V(n, m)` is represented as :math:`n \times m` matricies with orthogonal columns.
    """

    def __new__(cls, n: int, m: int, key, average_order: int = 1000):
        """
        :param n: the number of rows.
        :param m: the number of columns.
        :param key: random state used to sample from the stabilizer `SO(n-m)`.
        :param average_order: the number of random samples from the stabilizer `SO(n-m)`.
        :return: a tuple (new random state, a realization of `V(n, m)`).
        """

        assert n > m, "n should be greater than m"
        H = SOGroup(n - m)
        G = SOGroup(n)
        key, samples_H = H.random(key, average_order)
        new_space = super().__new__(cls)
        new_space.__init__(G=G, H=H, samples_H=samples_H, average_order=average_order)
        new_space.n = n
        new_space.m = m
        new_space.dim = G.dim - H.dim
        return key, new_space

    def project_to_manifold(self, g):
        """
        Take first m columns of an orthogonal matrix.

        :param g: [..., n, n] array of points in SO(n)
        :return: [..., n, m] array of points in V(n, m)
        """

        return g[..., : self.m]

    def embed_manifold(self, x):
        """
        Complete [n, m] matrix with orthogonal columns to an orthogonal [n, n] one using QR algorithm.

        :param x: [..., n, m] array of points in V(n, m)
        :return g: [..., n, n] array of points in SO(n)
        """

        g, r = qr(x, mode="complete")
        r_diag = einsum("...ii->...i", r[..., : self.m, : self.m])
        r_diag = B.concat(
            r_diag, B.ones(B.dtype(x), *x.shape[:-2], self.n - self.m), axis=-1
        )
        g = g * r_diag[..., None]
        diff = 2 * (B.all(B.abs(g[..., : self.m] - x) < 1e-5, axis=-1) - 0.5)
        g = g * diff[..., None]
        det_sign_g = B.sign(B.det(g))
        g[:, :, -1] *= det_sign_g[:, None]

        return g

    def embed_stabilizer(self, h):
        """
        Embed SO(n-m) matrix into SO(n) adding ones on diagonal,
        i.e. i(h) = [[h, 0], [0, 1]].

        :param h: [..., n-m, n-m] array of points in SO(n-m)
        :return res: [..., n, n] array of points in SO(n)
        """

        zeros = B.zeros(
            B.dtype(h), *h.shape[:-2], self.m, self.n - self.m
        )  # [m, n - m]
        zeros_t = B.transpose(zeros)  # [n - m, m]
        eye = B.tile(
            B.eye(B.dtype(h), self.m, self.m).reshape(
                *([1] * (len(h.shape) - 2)), self.m, self.m
            ),
            *h.shape[:-2],
            1,
            1,
        )  # [..., m, m]
        left = B.concat(eye, zeros_t, axis=-2)  # [..., n, m]
        right = B.concat(zeros, h, axis=-2)  # [..., n, n - m]
        res = B.concat(left, right, axis=-1)  # [..., n, n]
        return res

    def get_eigenfunctions(self, num: int) -> AveragingAdditionTheorem:
        eigenfunctions = StiefelEigenfunctions(self, num, self.samples_H)
        return eigenfunctions

    def get_repeated_eigenvalues(self, num: int) -> B.Numeric:
        return self.get_eigenvalues(num)

    def get_eigenvalues(self, num: int) -> B.Numeric:
        eigenfunctions = StiefelEigenfunctions(self, num, self.samples_H)
        eigenvalues = np.array(eigenfunctions._eigenvalues)
        return B.reshape(eigenvalues, -1, 1)  # [num, 1]
