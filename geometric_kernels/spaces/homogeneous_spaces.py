import lab as B
import json
from pathlib import Path
import more_itertools
import sympy
from sympy.matrices.determinant import _det as sp_det
import numpy as np
from geometric_kernels.lab_extras import dtype_double, from_numpy
from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import EigenfunctionWithAdditionTheorem, Eigenfunctions
from geometric_kernels.spaces.lie_groups import LieGroup
import abc


class HomogeneousSpaceAddtitionTheorem(EigenfunctionWithAdditionTheorem):

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

        diff = B.matmul(X_, X2_inv_).reshape(X.shape[0], X2_inv.shape[0], X.shape[-1], X.shape[-1] )  # [a,b, n, n]
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
    A class for Homogeneous Spaces represented as G/H, where G and H are groups.

    """
    def __init__(self, G: LieGroup, H, average_order=1000):
        self.G = G
        self.H = H
        self.dim = self.G.dim - self.H.dim
        self.average_order = average_order

    def H_to_G(self, h):
        """Implements inclusion H<G"""
        raise NotImplementedError

    def M_to_G(self, x):
        """Implements lifting M->G"""
        raise NotImplementedError

    def G_to_M(self, g):
        """Implements a canonical projection G->M"""
        raise NotImplementedError

    def sample_H(self, n):
        raw_samples = self.h.rand(n)
        return self.H_to_G(raw_samples)

    def pairwise_diff(self, x, y):
        """For arrays of form x_iH, y_jH computes difference Hx_i^{-1}y_jH """
        x_, y_ = self.M_to_G(x), self.M_to_G(y)
        diff = self.g.pairwise_diff(x_, y_, reverse=True)
        return diff

    def pairwise_embed(self, x, y):
        """For arrays of form x_iH, y_jH computes embedding corresponding to x_i, y_j
        i.e. flattened array of form G.embed(x_i^{-1}y_jh_k)"""
        x_y_ = self.pairwise_diff(x, y)
        return self.g.pairwise_embed(x_y_, self.h_samples)

    def compute_inv_dimension(self, signature):
        raise NotImplementedError


    @property
    def dimension(self) -> int:
        return self.dim

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        :param num: number of eigenfunctions returned.
        """
        return HomogeneousSpaceAddtitionTheorem(self.n, num)

    def get_eigenvalues(self, num: int) -> B.Numeric:
        """
        First `num` eigenvalues of the Laplace-Beltrami operator
        :return: [num, 1] array containing the eigenvalues
        """
        eigenfunctions = HomogeneousSpaceAddtitionTheorem(self.n, num)
        eigenvalues = np.array(eigenfunctions._eigenvalues)
        return B.reshape(eigenvalues, -1, 1)  # [num, 1]


    def random(self, key, number):
        key, raw_samples = self.g.rand(key, number)
        return key, self.G_to_M(raw_samples)
