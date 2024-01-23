import abc

import lab as B
import numpy as np
from opt_einsum import contract as einsum

from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import (
    Eigenfunctions,
    EigenfunctionWithAdditionTheorem,
)
from geometric_kernels.spaces.lie_groups import LieGroupCharacter, MatrixLieGroup


class AveragingAdditionTheorem(EigenfunctionWithAdditionTheorem):
    r"""
    Class corresponding to the sum of eigenfunctions corresponding
    to the same eigenspaces of Laplace-Beltrami operator on compact homogeneous space M=G/H
    Eigenspaces coincide with eigenspaces of G, and the sums might be computed via averaging
    of characters of group G w.r.t. H

    :math:`\chi_M(x) = \int_H \chi_G(xh)dh`
    """

    def __init__(self, M, num_levels, samples_H):
        """
        :param M: CompactHomogeneousSpace
        :param num_levels int: number of eigenspaces
        :param H_samples: samples from the uniform distribution on H
        """
        self.M = M
        self.dim = M.G.dim - M.H.dim
        self.G_n = M.G.n
        self.samples_H = samples_H
        self.samples_H = self.M.embed_stabilizer(samples_H)
        self.average_order = B.shape(samples_H)[0]

        G_eigenfunctions = M.G.get_eigenfunctions(num_levels)

        self._signatures = G_eigenfunctions._signatures.copy()
        self._eigenvalues = np.copy(G_eigenfunctions._eigenvalues)
        self._dimensions = np.copy(G_eigenfunctions._dimensions)
        self._characters = [
            AveragedLieGroupCharacter(self.average_order, character)
            for character in G_eigenfunctions._characters
        ]

        self.G_eigenfunctions = G_eigenfunctions
        self.G_torus_representative = G_eigenfunctions._torus_representative
        self.G_difference = G_eigenfunctions._difference

        self._num_levels = num_levels

    # @abc.abstractmethod
    def _compute_projected_character_value_at_e(self, signature):
        """
        Value of character on class of identity element is equal to the dimension of invariant space

        :param signature:
        :return: int
        """
        raise NotImplementedError

    def _difference(self, X: B.Numeric, X2: B.Numeric) -> B.Numeric:
        """
        Computes pairwise differences between points of the homogeneous space M embedded into G

        :param X: [N, ...] an array of points in M
        :param X2: [N2, ...] an array of points in M
        :return: [N, N2, ...] an array of points in G
        """

        g = self.M.embed_manifold(X)
        g2 = self.M.embed_manifold(X2)
        diff = self.G_difference(g, g2)
        return diff

    def _addition_theorem(self, X: B.Numeric, X2: B.Numeric, **parameters) -> B.Numeric:
        r"""
        Returns the result of applying the additional theorem when
        summing over all the eigenfunctions within a level, for each level

        To ensure that the resulting function is positive definite we average
        both left and right shifts

        :math:`\chi_X(g1,g2) \approx \frac{1}{S^2}\sum_{i=1}^S\sum_{j=1}^S \chi_G(h_i g2^{-1} g1 h_j)`
        :param X: [N, ...]
        :param X2: [N2, ...]
        :param parameters: unused.
        :return: Evaluate the sum of eigenfunctions on each level. Returns
            a value for each level [N, N2, L]
        """

        # [N * N2, G_n, G_n]
        diff = self._difference(X, X2).reshape(-1, self.G_n, self.G_n)
        # [N * N2 * samples_H, G_n, G_n]
        diff_h2 = self.G_difference(diff, self.samples_H).reshape(
            -1, self.G_n, self.G_n
        )
        # [H_samples * N * N2 * samples_H, G_n, G_n]
        h1_diff_h2 = self.G_difference(self.samples_H, diff_h2).reshape(
            -1, self.G_n, self.G_n
        )
        # [samples_H * N * N2 * samples_H, T]
        torus_repr = self.G_torus_representative(h1_diff_h2)
        values = [
            (degree * chi(torus_repr)[..., None]).reshape(
                X.shape[0], X2.shape[0], 1
            )  # [N1, N2, 1]
            for degree, chi in zip(self._dimensions, self._characters)
        ]

        return B.concat(*values, axis=-1)  # [N, N2, L]

    def _addition_theorem_diag(self, X: B.Numeric, **parameters) -> B.Numeric:
        """
        Returns the sum of eigenfunctions on a level for which we have a simplified expression

        :param X: [N, ...]
        :param parameters: any additional parameters
        :return: Evaluate the sum of eigenfunctions on each level. Returns
            a value for each level [N, L]
        """
        values = [
            degree * self._compute_projected_character_value_at_e(signature)  # [N, 1]
            for signature, degree in zip(self._signatures, self._dimensions)
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

    def __call__(self, X: B.Numeric):
        gammas = self._torus_representative(X)
        res = []
        for chi in self._characters:
            res.append(chi(gammas))
        res = B.stack(res, axis=1)
        return res


class AveragedLieGroupCharacter(abc.ABC):
    r"""
    Sum of eigenfunctions is equal to the mean value of the character averaged over subgroup H.
    To ensure that the function is positive definite we average from the both sides.

    :math:`\chi_M(x) = \int_H\int_H (h_1 x h_2)`
    """

    def __init__(self, average_order: int, character: LieGroupCharacter):
        """
        :param average_order: the number of points sampled from H
        param character: a character of a Lie group G.
        """
        self.character = character
        self.average_order = average_order

    def __call__(self, gammas_h1_x_h2):
        """
        Compute characters from the torus embedding and then averages w.r.t. H.
        :param gammas_h1_x_h2: [average_order*n*average_order, T]
        """
        character_h1_x_h2 = B.reshape(
            self.character(gammas_h1_x_h2), self.average_order, -1, self.average_order
        )
        avg_character = einsum("ugv->g", character_h1_x_h2) / (self.average_order**2)
        return avg_character


class CompactHomogeneousSpace(DiscreteSpectrumSpace):
    """
    Represents a compact homogeneous space X that are given as M=G/H,
    where G is a compact Lie group, H is a subgroup called stabilizer.

    Examples include Stiefel manifolds `SO(n) / SO(n-m)` and Grassmanians `SO(n)/(SO(m) x SO(n-m))`.
    """

    def __init__(self, G: MatrixLieGroup, H, samples_H, average_order):
        """
        :param G: A Lie group.
        :param H: stabilizer subgroup.
        :param samples_H: random samples from the stabilizer.
        :param average_order: average order.
        """
        self.G = G
        self.H = H
        self.samples_H = samples_H
        self.dim = self.G.dim - self.H.dim
        self.average_order = average_order

    @abc.abstractmethod
    def project_to_manifold(self, g):
        r"""
        Represents map \pi: G \mapsto X projecting elements of G onto X,
        e.g. \pi sends O \in SO(n) ([n,n] shape) to \pi(O) \in St(n,m) ([n,m] shape)
        by taking first m columns.

        :param g: [N, ...] array of points in G
        :return: [N, ...] array of points in M
        """
        raise NotImplementedError

    @abc.abstractmethod
    def embed_manifold(self, x):
        r"""
        Inverse to the function \pi, i.e. for given x \in X finds g such that \pi(g) = x.

        :param x: [N, ...] array of points in M
        :return: [N, ...] array of points in G
        """
        raise NotImplementedError

    @abc.abstractmethod
    def embed_stabilizer(self, h):
        """
        Embed subgroup H into G.

        :param h: [N, ...] array of points in H
        :return: [N, ...] array of points in G
        """
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        return self.dim

    def get_eigenfunctions(self, num: int, key) -> Eigenfunctions:
        """
        :param num: number of eigenfunctions returned.
        """
        eigenfunctions = AveragingAdditionTheorem(self, num, self.samples_H)
        return eigenfunctions

    def get_eigenvalues(self, num: int) -> B.Numeric:
        """
        Eigenvalues corresponding to the first `num` eigenspaces
        of the Laplace-Beltrami operator.

        :return: [num, 1] array containing the eigenvalues
        """
        eigenfunctions = AveragingAdditionTheorem(self, num, self.samples_H)
        eigenvalues = np.array(eigenfunctions._eigenvalues)
        return B.reshape(eigenvalues, -1, 1)  # [num, 1]

    def random(self, key, number: int):
        """
        Samples random points from the uniform distribution on M.

        :param key: a random state
        :param number: a number of random to generate
        :return [number, ...] an array of randomly generated points
        """
        key, raw_samples = self.g.rand(key, number)
        return key, self.project_to_manifold(raw_samples)
