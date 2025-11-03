r"""
Abstract base interface for compact homogeneous spaces.
"""

import abc

import lab as B
import numpy as np
from beartype.typing import List, Optional
from lab import einsum

from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import EigenfunctionsWithAdditionTheorem
from geometric_kernels.spaces.lie_groups import CompactMatrixLieGroup, LieGroupCharacter


class AveragingAdditionTheorem(EigenfunctionsWithAdditionTheorem):
    r"""
    Class corresponding to the sum of outer products of eigenfunctions
    corresponding to the same eigenspace of Laplace-Beltrami operator
    on a compact homogeneous space M=G/H. Eigenspaces coincide with
    eigenspaces of G, and the sums might be computed via averaging of
    characters of group G w.r.t. H.

    .. math:: \chi_M(x) = \int_H \chi_G(xh)dh

    .. note::
        *levels* here do not necessarily correspond to full eigenspaces.
    """

    def __init__(self, M, num_levels: int, samples_H):
        """
        :param M:
            CompactHomogeneousSpace.
        :param num_levels:
            Number of eigenspaces.
        :param samples_H:
            Samples from the uniform distribution on the stabilizer H.
        """
        self.M = M
        self.dim = M.dim
        self.G_n = M.G.n
        self.samples_H = samples_H
        self.samples_H = self.M.embed_stabilizer(samples_H)
        self.average_order = B.shape(samples_H)[0]

        G_eigenfunctions = M.G.get_eigenfunctions(num_levels)

        self._signatures = G_eigenfunctions._signatures.copy()
        self._eigenvalues = np.copy(G_eigenfunctions._eigenvalues)
        self.G_dimensions = G_eigenfunctions._dimensions.copy()
        self._characters = [
            AveragedLieGroupCharacter(self.average_order, character)
            for character in G_eigenfunctions._characters
        ]

        self._filter_signatures()

        print(
            f"Filtered out {len(G_eigenfunctions._signatures) - len(self._signatures)} eigenspaces of dimension 0."
        )
        self._num_levels = len(self._signatures)

        self.G_torus_representative = G_eigenfunctions._torus_representative
        self.G_difference = G_eigenfunctions._difference

        self._num_eigenfunctions: Optional[int] = None  # To be computed when needed.

    @abc.abstractmethod
    def _compute_projected_character_value_at_e(self, signature):
        r"""
        The value of the character on class of identity element.
        This is equal to the number of zonal spherical functions,
        it corresponds to $r_\lambda$ from :cite:t:`azangulov2024a`.

        :param signature:
            Signature of the character.

        :return:
            The value of the character on class of identity element.
        """
        raise NotImplementedError

    def _filter_signatures(self):
        """
        Filters out the eigenspaces of dimension 0.
        This is necessary to avoid numerical issues.
        Eigenspaces of dimension 0 correspond to
        the characters equal to 0 on the identity element.
        """
        filtered_signatures = []
        filtered_dimensions = []
        filtered_characters = []
        filtered_eigenvalues = []
        for signature, G_dimension, character, eigenvalue in zip(
            self._signatures, self.G_dimensions, self._characters, self._eigenvalues
        ):
            if self._compute_projected_character_value_at_e(signature) != 0:
                filtered_signatures.append(signature)
                filtered_dimensions.append(G_dimension)
                filtered_characters.append(character)
                filtered_eigenvalues.append(eigenvalue)
        self._signatures = filtered_signatures
        self.G_dimensions = filtered_dimensions
        self._characters = filtered_characters
        self._eigenvalues = np.array(filtered_eigenvalues)
        self._eigenspace_dimensions = [
            G_dim * self._compute_projected_character_value_at_e(sgn)
            for G_dim, sgn in zip(self.G_dimensions, self._signatures)
        ]
        self._num_zonal_spherical_functions = [
            self._compute_projected_character_value_at_e(sgn)
            for sgn in self._signatures
        ]

    def _difference(self, X: B.Numeric, X2: B.Numeric) -> B.Numeric:
        """
        Pairwise differences between points of the homogeneous space M
        embedded into G.

        :param X:
            [N1, ...] an array of points in `M`.
        :param X2:
            [N2, ...] an array of points in `M`.

        :return:
            [N1, N2, ...] an array of points in `G`.
        """

        g = self.M.embed_manifold(X)
        g2 = self.M.embed_manifold(X2)
        diff = self.G_difference(g, g2, inverse_X=True)
        return diff

    def _addition_theorem(
        self, X: B.Numeric, X2: Optional[B.Numeric] = None, **kwargs
    ) -> B.Numeric:
        r"""
        For each level (that corresponds to a unitary irreducible
        representation of the group of symmetries), computes the sum of outer
        products of Laplace-Beltrami eigenfunctions that correspond to this
        level (representation). Uses the fact that such a sum is always
        proportional to a certain integral of the character of the
        representation over the isotropy subgroup of the homogeneous space.
        See :cite:t:`azangulov2024a` for mathematical details.

        To ensure that the resulting function is positive definite, we average
        over both the left and right shifts (the result is an approximation):

        .. math:: \chi_X(g1,g2) \approx \frac{1}{S^2}\sum_{i=1}^S\sum_{j=1}^S \chi_G(h^{-1}_i g2^{-1} g1 h_j)

        :param X:
            An [N, n, m]-shaped array, a batch of N matrices of size nxm.
        :param X2:
            An [N2, n, m]-shaped array, a batch of N2 matrices of size nxm.
        :param ``**kwargs``:
            Any additional parameters

        :return:
            [N1, N2, L]
        """
        if X2 is None:
            X2 = X
        X_, X2_ = X, X2

        if X.shape[-1] != self.M.n:
            X_ = self.M.embed_manifold(X)
        if X2.shape[-1] != self.M.n:
            X2_ = self.M.embed_manifold(X2)

        # [N * N2, G_n, G_n]
        diff = self.G_difference(X_, X2_).reshape(-1, self.G_n, self.G_n)
        # [N * N2 * samples_H, G_n, G_n]
        diff_h = self.G_difference(diff, self.samples_H).reshape(-1, self.G_n, self.G_n)
        # [N * N2 * samples_H, T]
        torus_repr = self.G_torus_representative(diff_h)
        values = [
            (degree * chi(torus_repr)[..., None]).reshape(
                X.shape[0], X2.shape[0], 1
            )  # [N1, N2, 1]
            for degree, chi in zip(self.G_dimensions, self._characters)
        ]

        return B.concat(*values, axis=-1)  # [N, N2, L]

    def _addition_theorem_diag(self, X: B.Numeric, **parameters) -> B.Numeric:
        """
        A more efficient way of computing the diagonals of the matrices
        `self._addition_theorem(X, X)[:, :, l]` for all l from 0 to L-1.

        :param X:
            [N, ...]
        :param parameters:
            Any additional parameters.

        :return:
            [N, L]
        """
        ones = B.ones(B.dtype(X), *X.shape[:-2], 1)
        values = [
            eigenspace_dimension * ones  # [N, 1]
            for eigenspace_dimension in self._eigenspace_dimensions
        ]
        return B.concat(*values, axis=1)  # [N, L]

    @property
    def num_levels(self) -> int:
        """Number of levels, L"""
        return self._num_levels

    @property
    def num_eigenfunctions(self) -> int:
        if self._num_eigenfunctions is None:
            self._num_eigenfunctions = sum(self.num_eigenfunctions_per_level)
        return self._num_eigenfunctions

    @property
    def num_eigenfunctions_per_level(self) -> List[int]:
        """Number of eigenfunctions per level"""
        return self._eigenspace_dimensions


class AveragedLieGroupCharacter(abc.ABC):
    r"""
    Sum of outer products of eigenfunctions is equal to the mean value of the
    character averaged over the isotropy subgroup H. To ensure that the
    function is positive definite we average from the both sides.

    .. math:: \chi_M(x) = \int_H\int_H (h_1 x h_2).
    """

    def __init__(self, average_order: int, character: LieGroupCharacter):
        """
        :param average_order:
            The number of points sampled from H.
        :param character:
            A character of a Lie group G.
        """
        self.character = character
        self.average_order = average_order

    def __call__(self, gammas_x_h):
        """
        Compute characters from the torus embedding and then averages w.r.t. H.

        :param gammas_h1_x_h2:
            [average_order*n*average_order, T]
        """
        character_x_h = B.reshape(self.character(gammas_x_h), -1, self.average_order)
        avg_character = einsum("gv->g", character_x_h) / (self.average_order)
        return avg_character


class CompactHomogeneousSpace(DiscreteSpectrumSpace):
    """
    A compact homogeneous space M given as M=G/H, where G is a compact
    Lie group, H is a subgroup called the stabilizer.

    Examples include Stiefel manifolds SO(n) / SO(n-m) and
    Grassmannians SO(n)/(SO(m) x SO(n-m)).
    """

    def __init__(self, G: CompactMatrixLieGroup, dim_H, samples_H, average_order):
        """
        :param G:
            A Lie group.
        :param dim_H:
            Dimension of the stabilizer.
        :param samples_H:
            Random samples from the stabilizer.
        :param average_order:
            Average order.
        """
        self.G = G
        self.samples_H = samples_H
        self.dim = self.G.dimension - dim_H
        self.average_order = average_order

    @property
    def dimension(self) -> int:
        return self.dim

    @abc.abstractmethod
    def project_to_manifold(self, g):
        r"""
        Represents map $\pi: G \mapsto X$ projecting elements of G onto X,
        e.g. $\pi$ sends O \in SO(n) ([n,n] shape) to an element
        $\pi(O) \in V(m,n)$ ([n,m] shape) of the Stiefel manifold,
        by taking first m columns.

        :param g:
            [N, ...] array of points in G.
        :return:
            [N, ...] array of points in M.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def embed_manifold(self, x):
        r"""
        Inverse to the function $\pi$, i.e. for given $x \in X$ finds g
        (non-unique) such that $\pi(g) = x$.

        :param x:
            [N, ...] array of points in M.
        :return:
            [N, ...] array of points in G.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def embed_stabilizer(self, h):
        """
        Embed subgroup H into G.

        :param h:
            [N, ...] array of points in H.
        :return:
            [N, ...] array of points in G.
        """
        raise NotImplementedError

    def random(self, key, number: int, project=False):
        """
        Samples random points from the uniform distribution on M.

        :param key:
            A random state.
        :param number:
            A number of random to generate.

        :return:
            [number, ...] an array of randomly generated points.
        """
        key, raw_samples = self.G.random(key, number)
        if project:
            return key, self.project_to_manifold(raw_samples)
        else:
            return key, raw_samples
