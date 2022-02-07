"""
Special orthogonal group, SO(n)

The code is courtеsy of  Iskander Azangulov* и Andrei Smolensky*

* St. Petersburg University
"""

import itertools as it
import math
import operator
from functools import reduce

import geomstats as gs
import lab as B
import numpy as np

from geometric_kernels.eigenfunctions import Eigenfunctions
from geometric_kernels.lab_extras import from_numpy, swapaxes, take_along_axis
from geometric_kernels.spaces import DiscreteSpectrumSpace


def fixed_length_partitions(n, L):  # noqa
    """
    https://www.ics.uci.edu/~eppstein/PADS/IntegerPartitions.py
    Integer partitions of n into L parts, in colex order.
    The algorithm follows Knuth v4 fasc3 p38 in rough outline;
    Knuth credits it to Hindenburg, 1779.
    """

    # guard against special cases
    if L == 0:
        if n == 0:
            yield []
        return
    if L == 1:
        if n > 0:
            yield [n]
        return
    if n < L:
        return

    partition = [n - L + 1] + (L - 1) * [1]
    while True:
        yield partition.copy()
        if partition[0] - 1 > partition[1]:
            partition[0] -= 1
            partition[1] += 1
            continue
        j = 2
        s = partition[0] + partition[1] - 1
        while j < L and partition[j] >= partition[0] - 1:
            s += partition[j]
            j += 1
        if j >= L:
            return
        partition[j] = x = partition[j] + 1
        j -= 1
        while j > 0:
            partition[j] = x
            s -= x
            j -= 1
        partition[0] = s


class SOEigenfunctions(Eigenfunctions):
    """Eigenfunctions for Special Orthogonal Group SO(n)."""

    def __init__(self, dim: int, num_representations: int = 10):
        r"""
        :param dim: dimensionality in which the group acts.
        :param num_representation: number of representations of the group to compute. Values larger than 10 are impractical.
        """
        self.dim = dim  # In code referred to as D
        self.rank = dim // 2  # In code referred to as R
        self.num_representations = num_representations

        if self.dim % 2 == 0:
            self.rho = np.arange(self.rank)[::-1]
        else:
            self.rho = np.arange(self.rank)[::-1] + 0.5

        (
            self.signatures,
            self.repr_dims,
            self.repr_eigenvalues,
        ) = self._generate_signatures(self.num_representations)

    @property
    def num_eigenfunctions(self) -> int:
        """Number of eigenfunctions, M"""
        return self.signatures.shape[-1]

    def weighted_outerproduct(self, weights, X, X2, **parameters):
        r"""
        Computes :math:`\sum_{i=0}^{M-1} w_i \phi_i(x1) \phi_i(x2)`.
        Makes use of the fact that :math:`\phi_i(x1) \phi_i(x2) = \phi_i(x1 @ x2.T)`

        :param weights: [M, 1]
        :param X: Inputs where to evaluate the eigenfunctions, shape = [N, D, D]
            where `N` is the number of inputs and `D` the dimension.
        :param X2: Inputs where to evaluate the eigenfunctions, shape = [N2, D, D],
            where `N` is the number of inputs and `D` the dimension.
            Default to None, in which X is used for X2.
        :param parameters: any additional parameters

        :return: shape [N, N2]
        """
        if X2 is None:
            X2 = X

        x1x2T = B.einsum("nij,bkj->nbik", X, X2)

        Phi_prod = self.__call__(x1x2T)  # [N, N2, M]

        prod = B.sum(Phi_prod, weights, axis=-1)  # [N, N2]

        return prod

    def weighted_outerproduct_diag(self, weights, X, **parameters):
        r"""
        Computes :math:`\sum_{i=0}^{M-1} w_i \phi_i(x) \phi_i(x)`. Corresponds to the
        diagonal elements of `weighted_outproduct` but they can be calculated more
        efficiently.

        :param weights: [M, 1]
        :param X: Inputs where to evaluate the eigenfunctions, shape = [N, D, D].
        :param parameters: any additional parameters

        :return: shape [N, ]
        """
        x1x2T = B.einsum("nij,nkj->nik", X, X)

        Phi_prod = self.__call__(x1x2T)  # [N, M]

        prod = B.sum(Phi_prod * weights, axis=-1)  # [N, ]

        return prod  # [N, ]

    def __call__(self, X, **parameters):
        r"""
        Compute the eigenfunctions at `X`.

        :param X: [..., D, D] input matrices

        :return: [..., M] eigenfunctions
        """
        gamma = self.torus_embed(X)  # [..., R]

        chi = self.chi(gamma, from_numpy(X, self.signatures))  # [..., M]
        chi *= from_numpy(X, self.repr_dims)  # [..., M]

        return chi  # [..., M]

    def torus_embed(self, X):
        r"""
        :param X: [..., D, D]

        :return: [..., R]
        """
        eigv = B.eig(X, compute_eigvecs=False)  # [..., D]
        sorted_ind = B.argsort(np.real(eigv), axis=-1)  # [D, ]
        eigv = take_along_axis(eigv, sorted_ind, axis=-1)  # [..., D]
        gamma = eigv[..., 0:-1:2]  # [..., R]
        return gamma

    def xi0(self, qs, gamma):
        r"""
        :param qs: [M, R]
        :param gamma: [..., R]

        :return: [..., M]
        """
        gamma_expanded = B.expand_dims(
            gamma, (B.rank(gamma), B.rank(gamma) + 1)
        )  # [..., R, 1, 1]
        a = B.power(gamma_expanded, qs) + B.power(gamma_expanded, -qs)  # [..., R, M, R]
        a = swapaxes(a, -3, -2)  # [..., M, R, R]
        return B.det(a)  # [..., M]

    def xi1(self, qs, gamma):
        r"""
        :param qs: [M, R]
        :param gamma: [..., R]

        :return: [..., M]
        """
        gamma_expanded = B.expand_dims(
            gamma, (B.rank(gamma), B.rank(gamma) + 1)
        )  # [..., R, 1, 1]
        a = B.power(gamma_expanded, qs) - B.power(gamma_expanded, -qs)  # [..., R, M, R]
        a = swapaxes(a, -3, -2)  # [..., M, R, R]
        return B.det(a)  # [..., M]

    def chi(self, gamma, sgn):
        r"""
        :param gammma: [..., R]
        :param sgn: [M, R]

        :return: [..., M]
        """
        eps = 0.0
        gamma += eps  # [..., R]

        if self.dim % 2:
            qs = sgn + self.rank - B.range(B.shape(sgn)[-1]) - 1 / 2  # [M, R]
            ret = self.xi1(qs, gamma) / self.xi1(
                B.range(self.rank, 0, -1)[None, ::-1] - 1 / 2, gamma
            )  # [..., M]
            return ret
        else:
            qs = (
                sgn[:, ::-1] + self.rank - B.range(B.shape(sgn)[-1] - 1) - 1
            )  # [M, R - 1]
            qs = B.stack(qs, B.abs(sgn[:, self.rank - 1]))  # [M, R]
            # qs[:, self.rank - 1] = B.abs(sgn[:, self.rank - 1])
            if sgn[-1] == 0:
                return self.xi0(qs, gamma) / self.xi0(
                    B.range(self.rank)[None, ::-1], gamma
                )
            else:
                sign = B.sign(sgn[:, -1])
                return self.xi0(qs, gamma) + self.xi1(qs, gamma) * sign / self.xi0(
                    B.range(self.rank)[None, ::-1], gamma
                )

    def _generate_signatures(self, num_repr):
        r"""
        Generate signatures and dimensions of representations and the corresponding eigenvalues.
        """
        signatures = []
        for signature_sum in range(0, num_repr):
            for i in range(1, self.rank + 1):
                for signature in fixed_length_partitions(signature_sum, i):
                    signature.extend([0] * (self.rank - i))
                    signatures.append(tuple(signature))
                    if self.dim % 2 == 0 and signature[-1] != 0:
                        signature[-1] = -signature[-1]
                        signatures.append(tuple(signature))

        def _compute_dim(signature):
            if self.dim % 2 == 1:
                qs = [pk + self.rank - k - 1 / 2 for k, pk in enumerate(signature)]
                rep_dim = reduce(
                    operator.mul,
                    (
                        2 * qs[k] / math.factorial(2 * k + 1)
                        for k in range(0, self.rank)
                    ),
                ) * reduce(
                    operator.mul,
                    (
                        (qs[i] - qs[j]) * (qs[i] + qs[j])
                        for i, j in it.combinations(range(self.rank), 2)
                    ),
                    1,
                )
                return int(round(rep_dim))
            else:
                qs = [
                    pk + self.rank - k - 1 if k != self.rank - 1 else abs(pk)
                    for k, pk in enumerate(signature)
                ]
                rep_dim = int(
                    reduce(
                        operator.mul,
                        (2 / math.factorial(2 * k) for k in range(1, self.rank)),
                    )
                    * reduce(
                        operator.mul,
                        (
                            (qs[i] - qs[j]) * (qs[i] + qs[j])
                            for i, j in it.combinations(range(self.rank), 2)
                        ),
                        1,
                    )
                )
                return int(round(rep_dim))

        def _compute_eigenvalue(sgn):
            np_sgn = np.array(sgn)
            return np.norm(self.rho + np_sgn) ** 2 - np.norm(self.rho) ** 2

        signatures_vals = []
        for sgn in signatures:
            dim = _compute_dim(sgn)
            eigenvalue = _compute_eigenvalue(sgn)
            signatures_vals.append([sgn, dim, eigenvalue])

        signatures_vals.sort(key=lambda x: x[2])  # sort by eigenvalue
        signatures_vals = signatures_vals[:num_repr]

        signatures = np.array([x[0] for x in signatures_vals])  # [M, R]
        dims = np.array([x[1] for x in signatures_vals])  # [M, ]
        eigenvalues = np.array([x[2] for x in signatures_vals])  # [M, ]

        return signatures, dims, eigenvalues


class SpecialOrthogonalGroup(
    DiscreteSpectrumSpace, gs.geometry.special_orthogonal_group.SpecialOrthogonal
):
    r"""
    Special orthogonal group SO(n), that is a group of orthogonal matrices
    with determinant equal to 1.
    """

    def __init__(self, dim):
        super().__init__(self, dim=dim)
        self.cache = {}

    @property
    def dimension(self):
        return self.dim

    def get_eigenvalues(self, num: int):
        """
        First `num` eigenvalues.

        :param num: number of eigenvalues returned.

        :return: [num, 1] array containing the eigenvalues.
        """
        if num not in self.cache:
            eigenfunctions = SOEigenfunctions(self.dim, num)
            eigenvalues = eigenfunctions.repr_eigenvalues ** 2
            self.cache[num] = eigenfunctions
            return B.reshape(eigenvalues, -1, 1)
        else:
            eigenfunctions = self.cache[num]
            eigenvalues = eigenfunctions.repr_eigenvalues ** 2
            return B.reshape(eigenvalues, -1, 1)

    def get_eigenfunctions(self, num):
        """
        :param num: number of eigenfunctions returned.
        """
        if num not in self.cache:
            eigenfunctions = SOEigenfunctions(self.dim, num)
            self.cache[num] = eigenfunctions
            return eigenfunctions
        else:
            return self.cache[num]
