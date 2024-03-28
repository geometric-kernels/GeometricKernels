"""
Abstract base interface for compact matrix Lie groups.
"""
import abc

import lab as B
import numpy as np
from beartype.typing import List

from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import EigenfunctionWithAdditionTheorem


class WeylAdditionTheorem(EigenfunctionWithAdditionTheorem):
    r"""
    This implements the abstract base class for computing the sums of outer
    products of certain groups (we generally call "levels") of Laplace-Beltrami
    eigenfunctions on compact Lie groups. These are much like the *zonal
    spherical harmonics* of the :class:`SphericalHarmonics` class. However, the
    key to computing them is representation-theoretic: they are proportional to
    *characters* of irreducible unitary representations of the group. These
    characters, in their turn, can be algebraically computed using *Weyl
    character formula*. See [1] for the mathematical details behind this class.

    :param n:
        The order of the Lie group, e.g. for SO(5) this is 5, for SU(3) this is 3.
    :param num_levels:
        The number of levels used for kernel approximation. Here, each level
        corresponds to an irreducible unitary representation of the group.
    :param compute_characters:
        Whether or not to actually compute the *characters*. Setting this parameter
        to False might make sense if you do not care about eigenfunctions (or sums
        of outer products thereof), but care about eigenvalues, dimensions of
        irreducible unitary representations, etc. Defaults to True.

    **Note**: unlike :class:`SphericalHarmonics`, we do not expect the
    descendants of this class to compute the actual Laplace-Beltrami
    eigenfunctions (which are similar to (non-zonal) *spherical harmonics* of
    :class:`SphericalHarmonics`). In this case, it is not really necessary
    and computationally harder to do.

    **Note**: unlike in :class:`SphericalHarmonics`, here the levels do not
    necessarily correspond to whole eigenspaces (all eigenfunctions that
    correspond to the same eigenvalue). Here, levels are defined in terms of
    the algebraic structure of the group: they are the eigenfunctions that
    correspond to the same irreducible unitary representation of the group.

    References:

    [1] Iskander Azangulov, Andrei Smolensky, Alexander Terenin, Viacheslav
        Borovitskiy. Stationary Kernels and Gaussian Processes on Lie Groups
        and their Homogeneous Spaces I: the compact case.
    """

    def __init__(self, n: int, num_levels: int, compute_characters: bool = True):
        self._num_levels = num_levels
        self._signatures = self._generate_signatures(self._num_levels)
        self._eigenvalues = np.array(
            [self._compute_eigenvalue(signature) for signature in self._signatures]
        )
        self._dimensions = np.array(
            [self._compute_dimension(signature) for signature in self._signatures]
        )
        if compute_characters:
            self._characters = [
                self._compute_character(n, signature) for signature in self._signatures
            ]

    @abc.abstractmethod
    def _generate_signatures(self, num_levels: int):
        """
        Generate the signatures of `self.num_levels` irreducible unitary
        representations of the group that (likely) correspond to the smallest
        Laplace-Beltrami eigenvalues. These signatures index representations,
        mathematically they are vectors of dimension equal to the *rank* of the
        group, which in its turn is the dimension of maximal tori.

        :return:
            List of signatures of irreducible unitary representations. Each
            signature is itself a list of integers, whose length is equal to
            the rank of the group typically stored in `self.rank`.
        """

        """
        Generate signatures corresponding to `num_levels` representations.

        A signature determines a character (see `_compute_character` method) and an eigenvalue
        (`_compute_eigenvalue`).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_eigenvalue(self, signature: List[List[int]]):
        """
        Compute eigenvalue corresponding to `signature`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_dimension(self, signature: List[List[int]]):
        """
        Compute dimension of the representation corresponding to `signature`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_character(self, signature: List[List[int]]):
        """
        Compute character of the representation corresponding to `signature`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _torus_representative(self, X: B.Numeric):
        """
        The function maps a Lie group element `X` to a maximal torus of the
        Lie group.

        :param X: [N, D, D], a batch of N matrices of size DxD.
        :return: [N, R], a batch of N vectors of size R, where R is the
            dimension of maximal tori, i.e. the rank of the Lie group.
        """
        raise NotImplementedError

    def inverse(self, X: B.Numeric):
        """Computes inverse element in the group"""
        raise NotImplementedError

    def _difference(self, X: B.Numeric, X2: B.Numeric) -> B.Numeric:
        r"""
        Pairwise difference (in the group sense) between elements `X` and `X2`.

        :param X: [N1, D, D], a batch of N1 matrices of size DxD.
        :param X2: [N2, D, D], a batch of N2 matrices of size DxD.
        :return: [N1, N2, D, D], all pairwise "differences":
            X1[j, :, :] * inv(X2[i, :, :]) for all 0 <= i < N1, 0 <= j < N2.

        **Note**: doing X1[j, :, :] * inv(X2[i, :, :]) is as permissible as
            doing inv(X2[i, :, :]) * X1[j, :, :] which is actually used in [1].
            This is because :math:`\chi(x y x^{-1}) = \chi(y)` which implies
            that :math:`\chi(x y) = \chi(y x)`.
        """
        X2_inv = self.inverse(X2)
        X_ = B.tile(X[..., None, :, :], 1, X2_inv.shape[0], 1, 1)  # (N1, N2, D, D)
        X2_inv_ = B.tile(X2_inv[None, ..., :, :], X.shape[0], 1, 1, 1)  # (N1, N2, D, D)

        diff = B.matmul(X_, X2_inv_).reshape(
            X.shape[0], X2_inv.shape[0], X.shape[-1], X.shape[-1]
        )  # (N1, N2, D, D)
        return diff

    def _addition_theorem(self, X: B.Numeric, X2: B.Numeric, **parameters) -> B.Numeric:
        r"""
        For each level (that corresponds to a unitary irreducible
        representation of the group), computes the sum of outer products of
        Laplace-Beltrami eigenfunctions that correspond to this level
        (representation). Uses the fact that such sums are equal to the
        character of the representation multiplied by the dimension of
        that representation. See [1] for mathematical details.

        :param X: [N1, D, D]
        :param X2: [N2, D, D]
        :param parameters: any additional parameters
        :return: [N1, N2, L]
        """
        diff = self._difference(X, X2)
        torus_repr_diff = self._torus_representative(diff)
        values = [
            repr_dim * B.real(chi(torus_repr_diff)[..., None])  # [N1, N2, 1]
            for chi, repr_dim in zip(self._characters, self._dimensions)
        ]
        return B.concat(*values, axis=-1)  # [N1, N2, L]

    def _addition_theorem_diag(self, X: B.Numeric, **parameters) -> B.Numeric:
        """A more efficient way of computing the diagonals of the matrices
        `self._addition_theorem(X, X)[:, :, l]` for all l from 0 to L-1.

        :param X: [N, D, D]
        :param parameters: any additional parameters
        :return: [N, L]
        """
        ones = B.ones(B.dtype(X), *X.shape[:-2], 1)
        values = [
            repr_dim * repr_dim * ones  # [N, 1], because chi(X*inv(X))=repr_dim
            for repr_dim in self._dimensions
        ]
        return B.concat(*values, axis=1)  # [N, L]

    @property
    def num_levels(self) -> int:
        """Number of levels, L."""
        return self._num_levels

    @property
    def num_eigenfunctions(self) -> int:
        """Number of eigenfunctions, M."""
        return self._num_eigenfunctions

    @property
    def num_eigenfunctions_per_level(self) -> B.Numeric:
        """Number of eigenfunctions per level."""
        return self._dimensions**2

    def __call__(self, X: B.Numeric):
        raise NotImplementedError


class LieGroupCharacter(abc.ABC):
    """
    Class that represents a character of a Lie group.
    """

    @abc.abstractmethod
    def __call__(self, gammas):
        """
        Compute the character on `gammas` lying in a maximal torus.
        :param gammas: [..., rank] where `rank` is the dimension of a max-torus.
        """
        raise NotImplementedError


class MatrixLieGroup(DiscreteSpectrumSpace):
    r"""
    A base class for Lie groups of matrices, subgroups of the general linear
    group GL(n).

    The group operation is the standard matrix multiplication, and the group
    inverse is the standard matrix inverse. Despite this, we make the
    subclasses implement their own inverse routine, because in special cases
    it can typically be implemented much more efficient. For example, for the
    special orthogonal group :class:`SOGroup`, the inverse is equivalent to a
    simple transposition.
    """

    @property
    def dimension(self) -> int:
        return self.dim

    @abc.abstractmethod
    def inverse(self, X: B.Numeric) -> B.Numeric:
        """
        Inverse of the group element `X`.
        """
        raise NotImplementedError
