import abc
import lab as B
import numpy as np
from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import EigenfunctionWithAdditionTheorem, Eigenfunctions


class LieGroupAddtitionTheorem(EigenfunctionWithAdditionTheorem):
    def __init__(self, num_levels):
        self._num_levels = num_levels
        self._signatures, self._eigenvalues, self._dimensions = self._generate_signatures(self._num_levels)
        self._num_eigenfunctions = self.degree_to_num_eigenfunctions(self._num_levels)
        self._characters = [LieGroupCharacter(signature) for signature in self._signatures]

    def _generate_signatures(self, num_levels):
        raise NotImplementedError

    def _torus_representative(self, X):
        """The function maps Lie Group Element X to T -- a maximal torus of the Lie group
        [b, n, n] ---> [b, rank]"""
        raise NotImplementedError

    def inverse(self, X):
        """The function that computes inverse element in the group"""
        raise NotImplementedError

    def _difference(self, X: B.Numeric, X2: B.Numeric) -> B.Numeric:
        """X -- [a,n,n], X2 -- [b,n,n] --> [a,b,n,n]"""
        X2_inv = self.inverse(X2)
        X_ = B.tile(X[..., None, :, :], 1, X2_inv.shape[0], 1, 1)  # (a, b, n, n)
        X2_inv_ = B.tile(X2_inv[None, ..., :, :], X.shape[0], 1, 1, 1)  # (a, b, n, n)

        diff = B.matmul(X_, X2_inv_).reshape(X.shape[0], X2_inv.shape[0], X.shape[-1], X.shape[-1] )  # (a, b, n, n)
        return diff

    def _addition_theorem(self, X: B.Numeric, X2: B.Numeric, **parameters) -> B.Numeric:
        diff = self._difference(X, X2)
        torus_repr_diff = self._torus_representative(diff)
        values = [
            degree * chi(torus_repr_diff)[..., None]  # [N1, N2, 1]
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
        torus_repr_X = self._torus_representative(X)
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
        raise NotImplementedError

    def __call__(self, X: B.Numeric):
        gammas = self._torus_representative(X)
        res = []
        for chi in self._characters:
            res.append(chi(gammas))
        res = B.stack(res, axis=1)
        return res


class LieGroupCharacter(abc.ABC):
    def __call__(self, gammas):
        raise NotImplementedError


class LieGroup(DiscreteSpectrumSpace):
    r"""
    A class for Lie groups represented as matrices,
    """

    @property
    def dimension(self) -> int:
        return self.dim

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        :param num: number of eigenfunctions returned.
        """
        return LieGroupAddtitionTheorem(self.n, num)

    def get_eigenvalues(self, num: int) -> B.Numeric:
        """
        First `num` eigenvalues of the Laplace-Beltrami operator
        :return: [num, 1] array containing the eigenvalues
        """
        eigenfunctions = LieGroupAddtitionTheorem(self.n, num)
        eigenvalues = np.array(eigenfunctions._eigenvalues)
        return B.reshape(eigenvalues, -1, 1)  # [num, 1]

    def random(self, number):
        raise NotImplementedError