"""
Abstract base interface for compact matrix Lie groups and its abstract
friends, a subclass of :class:`~.eigenfunctions.Eigenfunctions` and an
abstract base class for Lie group characters.
"""

import abc

import lab as B
import numpy as np
from beartype.typing import List, Optional, Tuple

from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import EigenfunctionsWithAdditionTheorem


class LieGroupCharacter(abc.ABC):
    """
    Class that represents a character of a Lie group.
    """

    @abc.abstractmethod
    def __call__(self, gammas: B.Numeric) -> B.Numeric:
        """
        Compute the character on `gammas` lying in a maximal torus.

        :param gammas:
            [..., rank] where `rank` is the dimension of a maximal torus.

        :return:
            An array of shape [...] representing the values of the characters.
            The values can be complex-valued.
        """
        raise NotImplementedError


class WeylAdditionTheorem(EigenfunctionsWithAdditionTheorem):
    r"""
    A class for computing the sums of outer products of certain combinations
    (we call "levels") of Laplace-Beltrami eigenfunctions on compact Lie
    groups. These are much like the *zonal spherical harmonics* of the
    :class:`~.SphericalHarmonics` class. However, the key to computing them
    is representation-theoretic: they are proportional to *characters* of
    irreducible unitary representations of the group. These characters, in their
    turn, can be algebraically computed using the *Weyl character formula*. See
    :cite:t:`azangulov2024a` for the mathematical details behind this class.

    :param n:
        The order of the Lie group, e.g. for SO(5) this is 5, for SU(3) this is 3.
    :param num_levels:
        The number of levels used for kernel approximation. Here, each level
        corresponds to an irreducible unitary representation of the group.
    :param compute_characters:
        Whether or not to actually compute the *characters*. Setting this parameter
        to False might make sense if you do not care about eigenfunctions (or sums
        of outer products thereof), but care about eigenvalues, dimensions of
        irreducible unitary representations, etc.

        Defaults to True.

    .. note::
        Unlike :class:`~.SphericalHarmonics`, we do not expect the
        descendants of this class to compute the actual Laplace-Beltrami
        eigenfunctions (which are similar to (non-zonal) *spherical harmonics*
        of :class:`~.SphericalHarmonics`), implementing the `__call___` method.
        In this case, this is not really necessary and computationally harder.

    .. note::
        Unlike in :class:`~.SphericalHarmonics`, here the levels do not
        necessarily correspond to whole eigenspaces (all eigenfunctions that
        correspond to the same eigenvalue). Here, levels are defined in terms of
        the algebraic structure of the group: they are the eigenfunctions that
        correspond to the same irreducible unitary representation of the group.

    .. note::
        Here we break the general convention that the subclasses of the
        :class:`~.eigenfunctions.Eigenfunctions` only provide an interface for
        working with eigenfunctions, not eigenvalues, offering an interface
        for computing the latter as well.
    """

    def __init__(self, n: int, num_levels: int, compute_characters: bool = True):
        self._num_levels = num_levels
        self._signatures = self._generate_signatures(self._num_levels)
        self._eigenvalues = np.array(
            [self._compute_eigenvalue(signature) for signature in self._signatures]
        )
        self._dimensions = [
            self._compute_dimension(signature) for signature in self._signatures
        ]
        if compute_characters:
            self._characters = [
                self._compute_character(n, signature) for signature in self._signatures
            ]
        self._num_eigenfunctions: Optional[int] = None  # To be computed when needed.

    @abc.abstractmethod
    def _generate_signatures(self, num_levels: int) -> List[Tuple[int, ...]]:
        """
        Generate the signatures of `self.num_levels` irreducible unitary
        representations of the group that (likely) correspond to the smallest
        Laplace-Beltrami eigenvalues. These signatures index representations,
        mathematically they are vectors of dimension equal to the *rank* of the
        group, which in its turn is the dimension of maximal tori.

        :param num_levels:
            Number of levels.

        :return:
            List of signatures of irreducible unitary representations. Each
            signature is itself a tuple of integers, whose length is equal to
            the rank of the group typically stored in `self.rank`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_eigenvalue(self, signature: Tuple[int, ...]) -> B.Float:
        """
        Compute eigenvalue corresponding to `signature`.

        :param signature:
            The signature.

        :return:
            The eigenvalue.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_dimension(self, signature: Tuple[int, ...]) -> int:
        """
        Compute dimension of the representation corresponding to `signature`.

        :param signature:
            The signature.

        :return:
            The dimension.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_character(
        self, n: int, signature: Tuple[int, ...]
    ) -> LieGroupCharacter:
        """
        Compute character of the representation corresponding to `signature`.

        :param signature:
            The signature.

        :return:
            The character, represented by :class:`LieGroupCharacter`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _torus_representative(self, X: B.Numeric) -> B.Numeric:
        """
        The function maps a Lie group element `X` to a maximal torus of the
        Lie group.

        :param X:
            An [N, n, n]-shaped array, a batch of N matrices of size nxn.

        :return:
            An [N, R]-shape array, a batch of N vectors of size R, where R is
            the dimension of maximal tori, i.e. the rank of the Lie group.
        """
        raise NotImplementedError

    def inverse(self, X: B.Numeric) -> B.Numeric:
        """
        Should call the group's static
        :meth:`~.spaces.CompactMatrixLieGroup.inverse` method.

        :param X:
            As in :meth:`~.spaces.CompactMatrixLieGroup.inverse`.
        """
        raise NotImplementedError

    def _difference(self, X: B.Numeric, X2: B.Numeric) -> B.Numeric:
        r"""
        Pairwise difference (in the group sense) between elements of the
        two batches, `X` and `X2`.

        :param X:
            An [N, n, n]-shaped array, a batch of N matrices of size nxn.
        :param X2:
            An [N2, n, n]-shaped array, a batch of N2 matrices of size nxn.

        :return:
            An [N, N2, n, n]-shaped array, all pairwise "differences":
            X1[j, :, :] * inv(X2[i, :, :]) for all 0 <= i < N, 0 <= j < N2.

        .. note::
            Doing X1[j, :, :] * inv(X2[i, :, :]) is as permissible as
            doing inv(X2[i, :, :]) * X1[j, :, :] which is actually used in
            :cite:t:`azangulov2024a`. This is because $\chi(x y x^{-1})=\chi(y)$
            which implies that $\chi(x y) = \chi(y x)$.
        """
        X2_inv = self.inverse(X2)
        X_ = B.tile(X[..., None, :, :], 1, X2_inv.shape[0], 1, 1)  # (N, N2, n, n)
        X2_inv_ = B.tile(X2_inv[None, ..., :, :], X.shape[0], 1, 1, 1)  # (N, N2, n, n)

        diff = B.reshape(
            B.matmul(X_, X2_inv_), X.shape[0], X2_inv.shape[0], X.shape[-1], X.shape[-1]
        )  # (N, N2, n, n)
        return diff

    def _addition_theorem(
        self, X: B.Numeric, X2: Optional[B.Numeric] = None, **kwargs
    ) -> B.Numeric:
        r"""
        For each level (that corresponds to a unitary irreducible
        representation of the group), computes the sum of outer products of
        Laplace-Beltrami eigenfunctions that correspond to this level
        (representation). Uses the fact that such sums are equal to the
        character of the representation multiplied by the dimension of that
        representation. See :cite:t:`azangulov2024a` for mathematical details.

        :param X:
            An [N, n, n]-shaped array, a batch of N matrices of size nxn.
        :param X2:
            An [N2, n, n]-shaped array, a batch of N2 matrices of size nxn.

            Defaults to None, in which case X is used for X2.
        :param ``**kwargs``:
            Any additional parameters.

        :return:
            An array of shape [N, N2, L].
        """
        if X2 is None:
            X2 = X
        diff = self._difference(X, X2)
        torus_repr_diff = self._torus_representative(diff)
        values = [
            repr_dim * chi(torus_repr_diff)[..., None]  # [N, N2, 1]
            for chi, repr_dim in zip(self._characters, self._dimensions)
        ]
        return B.concat(*values, axis=-1)  # [N, N2, L]

    def _addition_theorem_diag(self, X: B.Numeric, **kwargs) -> B.Numeric:
        """
        A more efficient way of computing the diagonals of the matrices
        `self._addition_theorem(X, X)[:, :, l]` for all l from 0 to L-1.

        :param X:
            As in :meth:`_addition_theorem`.
        :param ``**kwargs``:
            As in :meth:`_addition_theorem`.

        :return:
            An array of shape [N, L].
        """
        ones = B.ones(B.dtype(X), *X.shape[:-2], 1)
        values = [
            repr_dim * repr_dim * ones  # [N, 1], because chi(X*inv(X))=repr_dim
            for repr_dim in self._dimensions
        ]
        return B.concat(*values, axis=1)  # [N, L]

    @property
    def num_levels(self) -> int:
        return self._num_levels

    @property
    def num_eigenfunctions(self) -> int:
        if self._num_eigenfunctions is None:
            self._num_eigenfunctions = sum(self.num_eigenfunctions_per_level)
        return self._num_eigenfunctions

    @property
    def num_eigenfunctions_per_level(self) -> List[int]:
        """
        The number of eigenfunctions per level.

        :return:
            List of squares of dimensions of the unitary irreducible
            representations.
        """
        return [d**2 for d in self._dimensions]


class CompactMatrixLieGroup(DiscreteSpectrumSpace):
    r"""
    A base class for compact Lie groups of matrices, subgroups of the
    real or complex general linear group GL(n), n being some integer.

    The group operation is the standard matrix multiplication, and the group
    inverse is the standard matrix inverse. Despite this, we make the
    subclasses implement their own inverse routine, because in special cases
    it can typically be implemented much more efficiently. For example, for
    the special orthogonal group :class:`~.spaces.SpecialOrthogonal`, the
    inverse is equivalent to a simple transposition.
    """

    @staticmethod
    @abc.abstractmethod
    def inverse(X: B.Numeric) -> B.Numeric:
        """
        Inverse of a batch `X` of the elements of the group. A static method.

        :param X:
            A batch [..., n, n] of elements of the group.
            Each element is a n x n matrix.

        :return:
            A batch [..., n, n] with each n x n matrix inverted.
        """
        raise NotImplementedError
