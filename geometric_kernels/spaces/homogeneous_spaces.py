import abc

import lab as B
import numpy as np

from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import (
    Eigenfunctions,
    EigenfunctionWithAdditionTheorem,
)
from geometric_kernels.spaces.lie_groups import LieGroupCharacter, MatrixLieGroup


class CompactHomogeneousSpaceAddtitionTheorem(EigenfunctionWithAdditionTheorem):
    def __init__(self, M, num_levels, H_samples):
        self.M = M
        self.dim = M.G.dim - M.H.dim
        self.G_n = M.G.n
        self.H_samples = H_samples
        self.H_samples = self.M.embed_stabilizer(H_samples)
        self.average_order = B.shape(H_samples)[0]
        G_eigenfunctions = M.G.get_eigenfunctions(num_levels)

        self.G_eigenfunctions = G_eigenfunctions

        self._signatures = G_eigenfunctions._signatures.copy()
        self._eigenvalues = np.copy(G_eigenfunctions._eigenvalues)
        self._dimensions = np.copy(G_eigenfunctions._dimensions)
        self._characters = [
            AveragedLieGroupCharacter(self.average_order, character)
            for character in G_eigenfunctions._characters
        ]
        self.G_torus_representative = G_eigenfunctions._torus_representative
        self.G_difference = G_eigenfunctions._difference
        self._num_levels = num_levels

    # @abc.abstractmethod
    def _compute_dimension(self, signature):
        raise NotImplementedError

    def _torus_representative(self, X):
        """The function maps Lie Group Element X to T -- a maximal torus of the Lie group
        [b, n, m] ---> [b, rank, h]"""
        return self.G_eigenfunctions._torus_representative(X)

    # @abc.abstractmethod
    def inverse(self, X):
        """The function that computes inverse element in the group"""
        raise NotImplementedError

    def _difference(self, X: B.Numeric, X2: B.Numeric) -> B.Numeric:
        """X -- [a,n,m], X2 -- [b,n,m] --> [a,b,n,n]"""
        g = self.M.embed_manifold(X)
        g2 = self.M.embed_manifold(X2)
        diff = self.G_difference(g, g2)
        return diff

    def _addition_theorem(self, X: B.Numeric, X2: B.Numeric, **parameters) -> B.Numeric:
        diff = self._difference(X, X2)
        diff = diff.reshape(X.shape[0] * X2.shape[0], self.G_n, self.G_n)
        diff_h = self.G_difference(diff, self.H_samples)
        torus_repr_diff = self.G_torus_representative(diff_h)
        values = [
            (degree * chi(torus_repr_diff)[..., None]).reshape(
                X.shape[0], X2.shape[0], 1
            )  # [N1, N2, 1]
            for degree, chi in zip(self._dimensions, self._characters)
        ]

        return B.concat(*values, axis=-1)  # [N, N2, L]

    def _addition_theorem_diag(self, X: B.Numeric, **parameters) -> B.Numeric:
        """
        Returns the sum of eigenfunctions on a level for which we have a simplified expression
        :param X: [N, D]
        :param parameters: any additional parameters
        :return: Evaluate the sum of eigenfunctions on each level. Returns
            a value for each level [N, L]
        """
        g = self.M.embed_manifold(X)
        g_h = self._difference(g, self.H_samples)
        torus_repr_X = self._torus_representative(g_h)
        values = [
            degree * chi(torus_repr_X)  # [N, 1]
            for chi, degree in zip(self._characters, self._dimensions)
        ]
        return B.concat(*values, axis=1)  # [N, L]

    @property
    def num_levels(self) -> int:
        """Number of levels, L"""
        return self._num_levels

    @property
    def num_eigenfunctions(self) -> int:
        """Number of eigenfunctions, M"""
        return self._num_eigenfunctions

    @property
    def num_eigenfunctions_per_level(self) -> B.Numeric:
        """Number of eigenfunctions per level"""
        return [1] * self.num_levels
        # raise NotImplementedError

    def __call__(self, X: B.Numeric):
        gammas = self._torus_representative(X)
        res = []
        for chi in self._characters:
            res.append(chi(gammas))
        res = B.stack(res, axis=1)
        return res


class AveragedLieGroupCharacter(abc.ABC):
    def __init__(self, average_order, character: LieGroupCharacter):
        self.character = character
        self.average_order = average_order

    def __call__(self, gammas_x_h):
        character_x_h = B.reshape(self.character(gammas_x_h), -1, self.average_order)
        avg_character_x = B.mean(character_x_h, axis=-1)
        return avg_character_x


class CompactHomogeneousSpace(DiscreteSpectrumSpace):
    r"""
    A Homogeneous Space of a Compact Lie Group, which is represented as M = G / H, where G is a Lie group,
    and H is a stabilizer subgroup.

    Examples of this class of spaces include Stiefel manifold (SO(n) / SO(n-m)).
    """

    def __init__(self, G: MatrixLieGroup, H, H_samples, average_order):
        """
        :param G: Lie group.
        :param H: stabilizer subgroup.
        :param H_samples: random samples from the stabilizer.
        :param average_order: average order.
        """
        self.G = G
        self.H = H
        self.H_samples = H_samples
        self.dim = self.G.dim - self.H.dim
        self.average_order = average_order

    @abc.abstractmethod
    def project_to_manifold(self, g):
        raise NotImplementedError

    @abc.abstractmethod
    def embed_manifold(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def embed_stabilizer(self, h):
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        return self.dim

    def get_eigenfunctions(self, num: int, key) -> Eigenfunctions:
        """
        :param num: number of eigenfunctions returned.
        """
        eigenfunctions = CompactHomogeneousSpaceAddtitionTheorem(
            self, num, self.H_samples
        )
        return eigenfunctions

    def get_eigenvalues(self, num: int) -> B.Numeric:
        """
        First `num` eigenvalues of the Laplace-Beltrami operator
        :return: [num, 1] array containing the eigenvalues
        """
        eigenfunctions = CompactHomogeneousSpaceAddtitionTheorem(
            self, num, self.H_samples
        )
        eigenvalues = np.array(eigenfunctions._eigenvalues)
        return B.reshape(eigenvalues, -1, 1)  # [num, 1]

    def random(self, key, number):
        key, raw_samples = self.g.rand(key, number)
        return key, self.project_to_manifold(raw_samples)
