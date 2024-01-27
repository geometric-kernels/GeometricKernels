import abc

import lab as B
import numpy as np

from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import EigenfunctionWithAdditionTheorem


class WeylAdditionTheorem(EigenfunctionWithAdditionTheorem):
    def __init__(self, n, num_levels, compute_characters=True):
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
    def _generate_signatures(self, num_levels):
        """
        Generate signatures corresponding to `num_levels` representations.

        A signature determines a character (see `_compute_character` method) and an eigenvalue
        (`_compute_eigenvalue`).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_eigenvalue(self, signature):
        """
        Compute eigenvalue corresponding to `signature`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_dimension(self, signature):
        """
        Compute dimension of the representation corresponding to `signature`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_character(self, signature):
        """
        Compute character of the representation corresponding to `signature`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _torus_representative(self, X):
        """The function maps a Lie Group element `X` to T -- a maximal torus of the Lie group.

        :param X: [b, n, n]
        :return: [b, rank]. `rank` is the dimension of maximal tori."""
        raise NotImplementedError

    def inverse(self, X):
        """Computes inverse element in the group"""
        raise NotImplementedError

    def _difference(self, X: B.Numeric, X2: B.Numeric) -> B.Numeric:
        """Pairwise difference (in the group sense) between elements `X` and `X2`.

        :param X: [a, n, n]
        :param X2: [b, n, n]
        :return: [a, b, n, n]"""
        X2_inv = self.inverse(X2)
        X_ = B.tile(X[..., None, :, :], 1, X2_inv.shape[0], 1, 1)  # (a, b, n, n)
        X2_inv_ = B.tile(X2_inv[None, ..., :, :], X.shape[0], 1, 1, 1)  # (a, b, n, n)

        diff = B.matmul(X_, X2_inv_).reshape(
            X.shape[0], X2_inv.shape[0], X.shape[-1], X.shape[-1]
        )  # (a, b, n, n)
        return diff

    def _addition_theorem(self, X: B.Numeric, X2: B.Numeric, **parameters) -> B.Numeric:
        diff = self._difference(X, X2)
        torus_repr_diff = self._torus_representative(diff)
        values = [
            degree**2 * chi(torus_repr_diff)[..., None]  # [N1, N2, 1]
            for chi, degree in zip(self._characters, self._dimensions)
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
        torus_repr_X = self._torus_representative(X)  # TODO: fixme
        values = [
            degree**2 * chi(torus_repr_X)  # [N, 1]
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
        return self._dimensions**2

    def __call__(self, X: B.Numeric):
        gammas = self._torus_representative(X)
        res = []
        for chi in self._characters:
            res.append(chi(gammas))
        res = B.stack(res, axis=1)
        return res


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
    A base class for Lie groups represented as matrices.
    """

    @property
    def dimension(self) -> int:
        return self.dim

    def get_eigenfunctions(self, num: int) -> WeylAdditionTheorem:
        """
        :param num: number of eigenfunctions returned.
        """
        raise NotImplementedError

    def get_eigenvalues(self, num: int) -> B.Numeric:
        """
        Eigenvalues corresponding to the first `num` levels.
        :return: [num, 1]  array containing the eigenvalues
        """
        eigenfunctions = WeylAdditionTheorem(self.n, num)
        eigenvalues = np.array(eigenfunctions._eigenvalues)
        return B.reshape(eigenvalues, -1, 1)  # [m, 1]

    def random(self, number):
        raise NotImplementedError
