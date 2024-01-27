import lab as B
import numpy as np
from opt_einsum import contract as einsum

from geometric_kernels.lab_extras import qr
from geometric_kernels.spaces.homogeneous_spaces import (
    AveragingAdditionTheorem,
    CompactHomogeneousSpace,
)
from geometric_kernels.spaces.so import SOGroup


class _SOxSO:
    """Helper class for sampling. Represents SO(n) x SO(m), as described by
    :math:`(n+m) \times (n+m)` block-diagonal matrices.
    """

    def __init__(self, n: int, m: int):
        self.n, self.m = n, m
        self.so_n = SOGroup(n)
        self.so_m = SOGroup(m)
        self.dim = self.so_n.dim + self.so_m.dim

    def random(self, key, number):
        """Randomly samples `number` matrices of size
        :math:`(n+m)\times(n+m)`.

        Each sample has a form of `[[H_n, 0], [0, H_m]]`.
        The upper left block is uniformly
        sampled over :math:`SO(n)`, and the lower right block is
        uniformly sampled over :math:`SO(m)`.
        """
        key, h_u = self.so_n.random(key, number)  # [number, n, n]
        key, h_d = self.so_m.random(key, number)  # [number, m, m]
        zeros = B.zeros(B.dtype(h_u), number, self.n, self.m)  # [number, n, m]
        zeros_t = B.transpose(zeros)

        # [number, n + m, n], [number, n + m, m]
        l, r = B.concat(h_u, zeros_t, axis=-2), B.concat(zeros, h_d, axis=-2)
        res = B.concat(l, r, axis=-1)  # [number, n + m, n + m]
        return key, res


class GrassmannianEigenfunctions(AveragingAdditionTheorem):
    def _compute_projected_character_value_at_e(self, signature) -> int:
        """
        Value of character on class of identity element is equal to the dimension of invariant space.
        In case of the Grassmannian this value always equal to 1, since the space is symmetric.

        :param signature: the signature of a representation.
        :return: value at e, the identity element.
        """

        return 1


class Grassmannian(CompactHomogeneousSpace):
    r"""
    Grassmanian manifold.

    Class for Grassmannian manifold :math:`Gr(n, m)` as
    :math:`SO(n) / (SO(m) \times SO(n-m))`.
    Elements of the manifold are represented as :math:`n \times m` matrices.
    """

    def __new__(cls, n, m, key, average_order=1000):
        """
        :param n: the number of rows.
        :param m: the number of coluns.
        :param key: random state used to sample from the stabilizer.
        :param average_order: the number of random samples from the stabilizer.
        :return: a tuple (new random state, a realization of `Gr(n, m)`).
        """

        assert n > m, "n should be greater than m"
        H = _SOxSO(m, n - m)
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

        :param x: [..., n, m] array of points in Gr(n, m)
        :return g: [..., n, n] array of points in SO(n)
        """

        g, r = qr(x, mode="complete")
        r_diag = einsum("...ii->...i", r[..., : self.m, : self.m])
        r_diag = B.concat(
            r_diag, B.ones(B.dtype(x), *x.shape[:-2], self.n - self.m), axis=-1
        )
        g = g * r_diag[:, None]
        diff = 2 * (B.all(B.abs(g[..., : self.m] - x) < 1e-5, axis=-1) - 0.5)
        g = g * diff[..., None]
        det_sign_g = B.sign(B.det(g))
        g[:, :, -1] *= det_sign_g[:, None]

        assert B.all((B.abs(x - g[:, :, : x.shape[-1]])) < 1e-6)
        return g

    def embed_stabilizer(self, h):
        """
        Embed SO(m) x SO(n-m) matrix into SO(n),
        In case of the Grassmannian, this is an identity mapping.

        :param h: [..., n, n] array of points in SO(m) x SO(n-m)
        :return: [..., n, n] array of points in SO(n)
        """
        return h

    def get_eigenfunctions(self, num: int) -> AveragingAdditionTheorem:
        eigenfunctions = GrassmannianEigenfunctions(self, num, self.samples_H)
        return eigenfunctions

    def get_repeated_eigenvalues(self, num: int) -> B.Numeric:
        return self.get_eigenvalues(num)

    def get_eigenvalues(self, num: int) -> B.Numeric:
        eigenfunctions = GrassmannianEigenfunctions(self, num, self.samples_H)
        eigenvalues = np.array(eigenfunctions._eigenvalues)
        return B.reshape(eigenvalues, -1, 1)  # [num, 1]
