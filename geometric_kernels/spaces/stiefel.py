"""
This module provides the :class:`Stiefel` space and the representation of
its spectrum, the :class:`StiefelEigenfunctions` class.
"""

from functools import lru_cache

import lab as B
import numpy as np

from geometric_kernels.lab_extras import dtype_double, qr
from geometric_kernels.spaces.homogeneous_spaces import (
    AveragingAdditionTheorem,
    CompactHomogeneousSpace,
)
from geometric_kernels.spaces.so import SpecialOrthogonal


def generate_intertwining_weights(n, omega):  # noqa: C901
    """
    Generate possible highest weights when branching from SO(n) to SO(n-1).
    omega: tuple representing the highest weight of SO(n).
    """
    mu = n // 2
    if n % 2 == 0:  # Even n = 2mu
        m_mu_abs = abs(omega[-1])

        def gen(k, prev):
            if k == mu - 1:  # omega' has mu - 1 components
                yield ()
            else:
                # Lower bound: |m_mu| for m_{mu-1}', else m_{k+1}
                lower = m_mu_abs if k == mu - 2 else omega[k + 1]
                upper = min(prev, omega[k])
                for m in range(lower, upper + 1):
                    for tail in gen(k + 1, m):
                        yield (m,) + tail

        for weight in gen(0, omega[0]):
            yield weight
    else:  # Odd n = 2mu + 1

        def gen(k, prev):
            if k == mu:  # omega' has mu components
                yield ()
            else:
                # Lower bound: |m_mu'| for last component, else m_{k+1}
                if k == mu - 1:
                    lower = -omega[k]  # m_mu >= |m_mu'|
                    upper = omega[k]
                else:
                    lower = omega[k + 1]
                    upper = min(prev, omega[k])
                for m in range(lower, upper + 1):
                    for tail in gen(k + 1, m):
                        yield (m,) + tail

        for weight in gen(0, omega[0]):
            yield weight


@lru_cache(maxsize=None)
def stiefel_multiplicity(n, m, omega):
    """
    Compute the multiplicity of the representation with highest weight omega
    in L^2(SO(n)/SO(n - m)).
    """
    if m == 0:
        # Base case: if omega is the trivial representation
        if all(x == 0 for x in omega):
            return 1
        return 0
    if n <= m:
        return 0  # Invalid case
    # Recursive case: branch from SO(n) to SO(n-1)
    total = 0
    for omega_prime in generate_intertwining_weights(n, omega):
        total += stiefel_multiplicity(n - 1, m - 1, omega_prime)
    return total


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

        return stiefel_multiplicity(self.M.n, self.M.m, signature)


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

        if n - m >= 2:
            H = SpecialOrthogonal(n - m)
            key, samples_H = H.random(key, average_order)
            dim_H = H.dim
        else:
            # H is a two point set {+1, -1}
            average_order = 2
            samples_H = B.zeros(dtype_double(key), average_order)
            samples_H[0], samples_H[1] = 1, -1
            samples_H = B.reshape(samples_H, average_order, 1, 1)
            dim_H = 0

        new_space = super().__new__(cls)
        key, matrix_complement = B.randn(
            key, B.dtype(samples_H), n, n - m
        )  # Shape: (n, n - m)

        new_space.__init__(G=G, dim_H=dim_H, samples_H=samples_H, average_order=average_order, n=n, m=m, matrix_complement=matrix_complement)  # type: ignore
        return key, new_space

    def __init__(
        self,
        G: SpecialOrthogonal,
        dim_H: int,
        samples_H: B.Numeric,
        matrix_complement: B.Numeric,
        average_order: int,
        n: int,
        m: int,
    ):
        self.n = n
        self.m = m
        self.matrix_complement = matrix_complement
        super().__init__(
            G=G, dim_H=dim_H, samples_H=samples_H, average_order=average_order
        )

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

        p = B.matmul(x, B.transpose(x, [0, 2, 1]))  # Shape: (b, n, n)
        r = self.matrix_complement[None, :, :]  # Shape: (1, n, n - m)

        r_orth = r - B.matmul(p, r)  # (b, n, n - m)

        q, _ = qr(r_orth)  # (b, n, n - m)

        g = B.concat(x, q, axis=2)  # (b, n, n)
        det = B.sign(B.det(g))
        g[:, :, -1] *= det[:, None]
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
        """
        Returns the :class:`~.AveragingAdditionTheorem` object with `num` levels.

        :param num:
            Number of levels.
        """
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
