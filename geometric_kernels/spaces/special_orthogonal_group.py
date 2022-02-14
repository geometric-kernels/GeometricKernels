"""
Special orthogonal group, SO(n)

The code is courtеsy of  Iskander Azangulov* и Andrei Smolensky*

* St. Petersburg University
"""

import itertools as it
import math
import operator
from functools import reduce
from typing import Any, Dict

import geomstats as gs
import lab as B
import numpy as np
from opt_einsum import contract as einsum

from geometric_kernels.eigenfunctions import Eigenfunctions
from geometric_kernels.lab_extras import (
    from_numpy,
    isclose,
    swapaxes,
    take_along_last_axis,
)
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
        assert (
            dim != 4
        ), "Dimension 4 is not supported since SO(4) is not a semisimple group."

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

        x1x2T = einsum("nij,mkj->nmik", X, X2)

        close_to_eye = self._close_to_eye(x1x2T)

        Phi_prod = self.__call__(x1x2T)  # [N, N2, M]

        prod = B.sum(Phi_prod * B.squeeze(weights), axis=-1)  # [N, N2]

        normalizing_constant = B.sum(
            from_numpy(X, B.squeeze(weights))
            * B.cast(B.dtype(X), from_numpy(X, self.repr_dims)) ** 2
        )

        output = B.where(close_to_eye, normalizing_constant, prod)  # [N, N2]

        return output / normalizing_constant

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
        x1x2T = einsum("nij,nkj->nik", X, X)

        close_to_eye = self._close_to_eye(x1x2T)  # [N, ]

        Phi_prod = self.__call__(x1x2T)  # [N, M]

        prod = B.sum(Phi_prod * B.squeeze(weights), axis=-1)  # [N, ]

        normalizing_constant = B.sum(
            from_numpy(X, B.squeeze(weights))
            * B.cast(B.dtype(X), from_numpy(X, self.repr_dims)) ** 2
        )

        output = B.where(close_to_eye, normalizing_constant, prod)  # [N, ]

        return output / normalizing_constant  # [N, ]

    def _close_to_eye(self, gram):
        r"""Check positions of the Gram matrix at which the product is near identity.

        :param gram: [..., D, D]
        :return: [..., ]
        """
        eye = B.eye(B.shape(gram)[-2], B.shape(gram)[-1])  # (D, D)
        eye = B.tile(eye, *(B.shape(gram)[:-2]), 1, 1)  # (..., D, D)
        eye = B.cast(B.dtype(gram), eye)

        close_to_eye = from_numpy(
            gram, B.all(B.all(isclose(gram, eye), axis=-1), axis=-1)
        )  # (...)

        return close_to_eye

    def __call__(self, X, **parameters):
        r"""
        Compute the eigenfunctions at `X`.

        :param X: [..., D, D] input matrices

        :return: [..., M] eigenfunctions
        """
        gamma = self.torus_embed(X)  # [..., R]
        chi = self.chi(gamma, from_numpy(X, self.signatures))  # [..., M]
        chi *= B.cast(B.dtype(chi), from_numpy(X, self.repr_dims))  # [..., M]

        return B.real(chi)  # [..., M]

    def torus_embed(self, X):
        r"""
        :param X: [..., D, D]

        :return: [..., R]
        """
        eigv = B.eig(X, compute_eigvecs=False)  # [..., D]
        sorted_ind = B.argsort(B.real(eigv), axis=-1)  # [..., D ]
        eigv = take_along_last_axis(eigv, sorted_ind)  # [..., D]
        gamma = eigv[..., 0:-1:2]  # [..., R]
        return gamma

    def xi0(self, qs, gamma):
        r"""
        :param qs: [M, R]
        :param gamma: [..., R]

        :return: [..., M]
        """
        qs = B.cast(B.dtype(gamma), qs)
        gamma_expanded = B.expand_dims(
            B.expand_dims(gamma, B.rank(gamma)), B.rank(gamma) + 1
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
        qs = B.cast(B.dtype(gamma), qs)
        gamma_expanded = B.expand_dims(
            B.expand_dims(gamma, B.rank(gamma)), B.rank(gamma) + 1
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
            qs = (
                B.cast(B.dtype_float(sgn), sgn)
                + B.cast(B.dtype_float(self.rank), self.rank)
                - B.range(
                    B.dtype_float(sgn),
                    B.shape(sgn)[-1],
                )
                - 1 / 2
            )  # [M, R]
            ret = self.xi1(qs, gamma) / self.xi1(
                B.range(self.rank, 0, -1)[None, :] - 1 / 2, gamma
            )  # [..., M]
            return ret
        else:
            qs = (
                B.cast(B.dtype_float(sgn), sgn[:, :-1])
                + B.cast(B.dtype_float(self.rank), self.rank)
                - B.range(B.dtype_float(sgn), B.shape(sgn)[-1] - 1)
                - 1
            )  # [M, R - 1]
            qs = B.concat(
                qs, B.cast(B.dtype(qs), B.abs(sgn[:, None, self.rank - 1])), axis=1
            )  # [M, R]
            ret = B.where(
                sgn[:, -1] == 0,
                self.xi0(qs, gamma)
                / self.xi0(B.range(B.dtype(qs), self.rank - 1, -1, -1)[None, :], gamma),
                (
                    self.xi0(qs, gamma)
                    + self.xi1(qs, gamma) * B.cast(B.dtype(gamma), B.sign(sgn[:, -1]))
                )
                / self.xi0(B.range(B.dtype(qs), self.rank - 1, -1, -1)[None, :], gamma),
            )
            return ret

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
            return (
                np.linalg.norm(self.rho + np_sgn) ** 2 - np.linalg.norm(self.rho) ** 2
            )

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
    DiscreteSpectrumSpace, gs.geometry.special_orthogonal._SpecialOrthogonalMatrices
):
    r"""
    Special orthogonal group `SO(n)`, that is a group of orthogonal matrices
    with determinant equal to 1.
    """

    def __init__(self, n: int):
        r"""
        :param n: Dimensionality `SO(n)`, aka the size of the matrices.
        """
        self.cache: Dict[int, Any] = {}
        super().__init__(n=n)

    @property
    def dimension(self):
        return self.n

    def get_eigenvalues(self, num: int):
        """
        First `num` eigenvalues.

        :param num: number of eigenvalues returned.

        :return: [num, 1] array containing the eigenvalues.
        """
        if num not in self.cache:
            eigenfunctions = SOEigenfunctions(self.n, num)
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
