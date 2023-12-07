import lab as B
import numpy as np
from opt_einsum import contract as einsum

from geometric_kernels.lab_extras import qr
from geometric_kernels.spaces.homogeneous_spaces import (
    CompactHomogeneousSpace,
    CompactHomogeneousSpaceAddtitionTheorem,
)
from geometric_kernels.spaces.so import SOGroup


class _SOxSO:
    """Helper class for sampling, represents SO(n) x SO(m)"""

    def __init__(self, n, m):
        self.n, self.m = n, m
        self.so_n = SOGroup(n)
        self.so_m = SOGroup(m)
        self.dim = n * (n - 1) // 2 + m * (m - 1) // 2

    def random(self, key, number):
        key, h_u = self.so_n.random(key, number)
        key, h_d = self.so_m.random(key, number)
        zeros = B.zeros(B.dtype(h_u), number, self.n, self.m)
        zeros_t = B.transpose(zeros)
        l, r = B.concat(h_u, zeros_t, axis=-2), B.concat(zeros, h_d, axis=-2)
        res = B.concat(l, r, axis=-1)
        return key, res


class GrassmannianEigenfunctions(CompactHomogeneousSpaceAddtitionTheorem):
    def _compute_projected_character_value_at_e(self, signature):
        """
        Value of character on class of identity element is equal to the dimension of invariant space
        In case of grassmannian this value always equal to 1, since the space is symmetric.
        :param signature:
        :return: int
        """
        return 1


class Grassmannian(CompactHomogeneousSpace):
    r"""
    Class for Grassmannian manifold as SO(n)/(SO(m)xSO(n-m))
    Elements of manifold represented as nxm matrices, note that
    X and X x (SO(m) \oplus I_{n-m}) are representatives of the same element
    """

    def __new__(cls, n, m, key, average_order=1000):
        assert n > m, "n should be greater than m"
        H = _SOxSO(m, n - m)
        G = SOGroup(n)
        key, H_samples = H.random(key, average_order)
        new_space = super().__new__(cls)
        new_space.__init__(G=G, H=H, H_samples=H_samples, average_order=average_order)
        new_space.n = n
        new_space.m = m
        new_space.dim = G.dim - H.dim
        return key, new_space

    def project_to_manifold(self, g):
        return g[..., : self.m]

    def embed_manifold(self, x):
        g, r = qr(x)
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
        return h

    @property
    def dimension(self) -> int:
        return self.dim

    def get_eigenfunctions(self, num: int) -> CompactHomogeneousSpaceAddtitionTheorem:
        """
        :param num: number of eigenfunctions returned.
        """
        eigenfunctions = GrassmannianEigenfunctions(self, num, self.H_samples)
        return eigenfunctions

    def get_repeated_eigenvalues(self, num: int) -> B.Numeric:
        pass

    def get_eigenvalues(self, num: int) -> B.Numeric:
        """
        First `num` eigenvalues of the Laplace-Beltrami operator
        :return: [num, 1] array containing the eigenvalues
        """
        eigenfunctions = GrassmannianEigenfunctions(self, num, self.H_samples)
        eigenvalues = np.array(eigenfunctions._eigenvalues)
        return B.reshape(eigenvalues, -1, 1)  # [num, 1]

    def random(self, key, number):
        key, raw_samples = self.G.random(key, number)
        return key, self.project_to_manifold(raw_samples)
