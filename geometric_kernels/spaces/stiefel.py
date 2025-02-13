"""
This module provides the :class:`Stiefel` space and the representation of
its spectrum, the :class:`StiefelEigenfunctions` class.
"""

import lab as B
import numpy as np
import math
from opt_einsum import contract as einsum

from geometric_kernels.lab_extras import qr, dtype_double
from geometric_kernels.spaces.homogeneous_spaces import (
    AveragingAdditionTheorem,
    CompactHomogeneousSpace,
)
from geometric_kernels.spaces.so import SpecialOrthogonal


def _hook_content_formula(lmd, n):
    """
    A combinatorial formula used to calculate the dimension of invariant space
    of the Stiefeld manifold (among other things).
    """
    numer = 1
    denom = 1

    l_cols = [sum([row_l >= i + 1 for row_l in lmd]) for i in range(lmd[0])]
    for id_row, l_row in enumerate(lmd):
        for id_col in range(l_row):
            numer *= n + id_col - id_row
            denom *= l_cols[id_col] + l_row - id_row - id_col - 1

    return numer // denom

def sample_SO2(key, number):
        key, thetas = B.random.randn(key, dtype_double(key), number, 1)
        thetas = 2 * math.pi * thetas
        c = B.cos(thetas)
        s = B.sin(thetas)
        r1 = B.stack(c, s, axis=-1)
        r2 = B.stack(-s, c, axis=-1)
        q = B.concat(r1, r2, axis=-2)
        return key, q

class StiefelEigenfunctions(AveragingAdditionTheorem):
    def _compute_projected_character_value_at_e(self, signature):
        """
        Value of character on the class of identity element is equal to the
        dimension of invariant space. In case of Stiefel manifold it could be
        computed using the hook-content formula.

        :param signature:
            The character signature.

        :return:
            Value of character on the class of identity element.
        """

        m_ = min(self.M.m, self.M.n - self.M.m)
        if m_ < self.M.G.rank and signature[m_] > 0:
            return 0
        signature_abs = tuple(abs(x) for x in signature)
        return _hook_content_formula(signature_abs, m_)


class Stiefel(CompactHomogeneousSpace):
    r"""
    The GeometricKernels space representing the Stifiel manifold
    V(m, n) as the homogeneous space SO(n) / SO(n-m).

    The elements of this space are represented as n x m matrices
    with orthogonal columns.

    .. note::
        A tutorial on how to use this space is available in the
        :doc:`Stiefel.ipynb </examples/Stiefel>` notebook.

    .. admonition:: Citation

        If you use this GeometricKernels space in your research, please consider
        citing :cite:t:`azangulov2022`.
    """

    def __new__(cls, n: int, m: int, key, average_order: int = 100):
        """
        :param n:
            The number of rows.
        :param m:
            The number of columns.
        :param key:
            Random state used to sample from the stabilizer SO(n-m).
        :param average_order:
            The number of random samples from the stabilizer SO(n-m).

        :return:
            A tuple (new random state, a realization of V(m, n)).
        """

        assert n > m, "n should be greater than m"
        G = SpecialOrthogonal(n)
        if n-m >= 3:
            H = SpecialOrthogonal(n - m)
            key, samples_H = H.random(key, average_order)
            dim_H = H.dim
        elif n-m == 2:
            # H is a circle
            key, samples_H = sample_SO2(key, average_order)
            dim_H = 1
        else:
            # H is a two point set {+1, -1}
            average_order = 2
            samples_H = B.zeros(dtype_double(key), average_order)
            samples_H[0], samples_H[1] = 1, -1
            samples_H = B.reshape(samples_H, average_order, 1, 1)
            dim_H = 0

        new_space = super().__new__(cls)
        new_space.__init__(G=G, dim_H=dim_H, samples_H=samples_H, average_order=average_order, n=n, m=m)  # type: ignore
        return key, new_space

    def __init__(
        self,
        G: SpecialOrthogonal,
        dim_H: int,
        samples_H: B.Numeric,
        average_order: int,
        n: int,
        m: int,
    ):
        super().__init__(G=G, dim_H=dim_H, samples_H=samples_H, average_order=average_order)
        self.n = n
        self.m = m

    def project_to_manifold(self, g):
        """
        Take first m columns of an orthogonal matrix.

        :param g:
            [..., n, n] array of points in SO(n).

        :return:
            [..., n, m] array of points in V(m, n).
        """

        return g[..., : self.m]

    def embed_manifold(self, x):
        """
        Complete [n, m] matrix with orthogonal columns to an orthogonal
        [n, n] one using QR algorithm.

        :param x:
            [..., n, m] array of points in V(m, n).

        :return g:
            [..., n, n] array of points in SO(n).
        """

        g, r = qr(x, mode="complete")
        r_diag = einsum("...ii->...i", r[..., : self.m, : self.m])
        r_diag = B.concat(
            r_diag, B.ones(B.dtype(x), *x.shape[:-2], self.n - self.m), axis=-1
        )
        g = g * r_diag[..., None]
        diff = 2 * (1.0 * B.all(B.abs(g[..., : self.m] - x) < 1e-5, axis=-1) - 0.5)
        g = g * diff[..., None]
        det_sign_g = B.sign(B.det(g))
        g[:, :, -1] *= det_sign_g[:, None]

        return g

    def embed_stabilizer(self, h):
        """
        Embed SO(n-m) matrix into SO(n) adding ones on diagonal,
        i.e. i(h) = [[h, 0], [0, 1]].

        :param h:
            [..., n-m, n-m] array of points in SO(n-m).

        :return:
            [..., n, n] array of points in SO(n).
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

    @property
    def element_shape(self):
        """
        Return the shape of the elements of the Stiefel manifold.
        """
        return [self.n, self.m]
    
    @property
    def element_dtype(self):
        """
        Return the data type of the elements of the Stiefel manifold.
        """
        return B.Float

    @property
    def dimension(self):
        """
        Return the dimension of the Stiefel manifold.
        """
        return self.n * self.m - self.m * (self.m + 1) // 2

