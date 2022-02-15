"""
Spaces for which there exist analytical expressions for the manifold
and the eigenvalues and functions. Examples include the `Circle` and the `Hypersphere`.
The Geomstats package is used for most of the geometric calculations.
"""

from typing import Tuple

import geomstats as gs
import lab as B
from spherical_harmonics import SphericalHarmonics as _SphericalHarmonics
from spherical_harmonics.fundamental_set import num_harmonics

from geometric_kernels.eigenfunctions import (
    Eigenfunctions,
    EigenfunctionWithAdditionTheorem,
)
from geometric_kernels.spaces import DiscreteSpectrumSpace
from geometric_kernels.utils import Optional, chain


class SphericalHarmonics(EigenfunctionWithAdditionTheorem):
    """
    Eigenfunctions Laplace-Beltrami operator on the sphere correspond to the
    spherical harmonics.
    """

    def __init__(self, dim: int, num_eigenfunctions: int) -> None:
        """
        :param dim:
            S^{dim}. Example: dim = 2 is the sphere in R^3. Follows geomstats convention.

        :param num_eigenfunctions:
            Specifies the minimum degree of the spherical harmonic collection so that
            there are at least `num_eigenfunctions`.
        """
        self.dim = dim
        self._num_levels, self._num_eigenfunctions = self.num_eigenfunctions_to_degree(
            num_eigenfunctions
        )
        self._spherical_harmonics = _SphericalHarmonics(
            dimension=dim + 1, degrees=self._num_levels
        )

    def num_eigenfunctions_to_degree(self, num_eigenfunctions: int) -> Tuple[int, int]:
        """
        Returns the minimum degree for which there are at least
        `num_eigenfunctions` in the collection.
        """
        n, d = 0, 0
        while n < num_eigenfunctions:
            n += num_harmonics(self.dim + 1, d)
            d += 1

        if n > num_eigenfunctions:
            print(
                "The number of eigenfunctions requested does not lead to complete "
                "levels of spherical harmonics. We have thus increased the number to "
                f"{n}, which includes all spherical harmonics up to degree {d} (excl.)"
            )
        return d, n

    def __call__(self, X: B.Numeric, **parameters) -> B.Numeric:
        r"""
        Evaluates the spherical harmonics at `X`, which are Euclidian coordinates.
        In other words, the points are parameterized by their extrinsic
        (self.dim+1)-coordinates.

        :param X: TensorType, [N, self.dim+1]
            N points with unit norm in Euclidian coordinate system (extrinsic).

        :return: [N, M], where M = self.num_eigenfunctions
        """
        return self._spherical_harmonics(X)

    def _addition_theorem(self, X: B.Numeric, X2: B.Numeric, **parameters) -> B.Numeric:
        r"""
        Returns the result of applying the additional theorem when
        summing over all the eigenfunctions within a level, for each level

        Concretely, in the case for inputs on the hypersphere, summing over all the
        spherical harmonics within a level is equal to evaluating the Gegenbauer polynomial.

        :param X: [N, dim+1]
        :param X2: [N2, dim+1]
        :param parameters: unused.
        :return: Evaluate the sum of eigenfunctions on each level. Returns
            a value for each level [N, N2, L]
        """
        values = [
            level.addition(X, X2)[..., None]  # [N1, N2, 1]
            for level in self._spherical_harmonics.harmonic_levels
        ]
        return B.concat(*values, axis=2)  # [N, N2, L]

    def _addition_theorem_diag(self, X: B.Numeric, **parameters) -> B.Numeric:
        """
        Returns the sum of eigenfunctions on a level for which we have a simplified expression

        :param X: [N, 1]
        :param parameters: unused.
        :return: Evaluate the sum of eigenfunctions on each level. Returns
            a value for each level [N, L]
        """
        values = [
            level.addition_at_1(X)  # [N, 1]
            for level in self._spherical_harmonics.harmonic_levels
        ]
        return B.concat(*values, axis=1)  # [N, L]

    @property
    def num_eigenfunctions(self) -> int:
        """Number of eigenfunctions, M"""
        return self._num_eigenfunctions

    @property
    def num_levels(self) -> int:
        """
        Number of levels, L

        For each level except the first where there is just one, there are two
        eigenfunctions.
        """
        return self._num_levels

    @property
    def num_eigenfunctions_per_level(self) -> B.Numeric:
        """Number of eigenfunctions per level, [N_l]_{l=0}^{L-1}"""
        return [num_harmonics(self.dim + 1, level) for level in range(self.num_levels)]


class Hypersphere(DiscreteSpectrumSpace, gs.geometry.hypersphere.Hypersphere):
    r"""
    The d-dimensional hypersphere embedded in the (d+1)-dimensional Euclidean space.
    """

    def __init__(self, dim: int):
        r"""
        :param dim: Dimension of the hypersphere :math:`S^d`.
        """
        super().__init__(dim=dim)
        self.dim = dim

    def is_tangent(
        self,
        vector: B.Numeric,
        base_point: Optional[B.Numeric] = None,  # type: ignore
        atol: float = gs.backend.atol,
    ) -> bool:
        """
        Check whether the `vector` is tangent at `base_point`.
        :param vector: shape=[..., dim]
            Vector to evaluate.
        :param base_point: shape=[..., dim]
            Point on the manifold. Defaults to `None`.
        :param atol: float
            Absolute tolerance.
            Optional, default: 1e-6.
        :return: Boolean denoting if vector is a tangent vector at the base point.
        """
        raise NotImplementedError("`is_tangent` is not implemented for `Hypersphere`")

    @property
    def dimension(self) -> int:
        return self.dim

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        :param num: number of eigenfunctions returned.
        """
        return SphericalHarmonics(self.dim, num)

    def get_eigenvalues(self, num: int) -> B.Numeric:
        """
        First `num` eigenvalues of the Laplace-Beltrami operator

        :return: [num, 1] array containing the eigenvalues
        """
        eigenfunctions = SphericalHarmonics(self.dim, num)
        eigenvalues_per_level = [
            level.eigenvalue()
            for level in eigenfunctions._spherical_harmonics.harmonic_levels
        ]
        eigenvalues = chain(
            eigenvalues_per_level,
            eigenfunctions.num_eigenfunctions_per_level,
        )  # [num,]
        return B.reshape(eigenvalues, -1, 1)  # [num, 1]
