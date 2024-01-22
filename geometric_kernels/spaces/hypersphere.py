"""
Spaces for which there exist analytical expressions for the manifold
and the eigenvalues and functions. Examples include the `Circle` and the `Hypersphere`.
The Geomstats package is used for most of the geometric calculations.
"""

from typing import Tuple

import geomstats as gs
import lab as B
import numpy as np
from spherical_harmonics import SphericalHarmonics as _SphericalHarmonics
from spherical_harmonics.fundamental_set import num_harmonics

from geometric_kernels.lab_extras import dtype_double
from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import (
    Eigenfunctions,
    EigenfunctionWithAdditionTheorem,
)
from geometric_kernels.utils.utils import chain


class SphericalHarmonics(EigenfunctionWithAdditionTheorem):
    """
    Eigenfunctions Laplace-Beltrami operator on the sphere correspond to the
    spherical harmonics.
    """

    def __init__(self, dim: int, num_levels: int) -> None:
        """
        :param dim:
            S^{dim}. Example: dim = 2 is the sphere in R^3. Follows geomstats convention.

        :param num_levels:
            Specifies the number of levels (degrees) of the spherical harmonics.
        """
        self.dim = dim
        self._num_levels = num_levels
        self._num_eigenfunctions = self.degree_to_num_eigenfunctions(self._num_levels)
        self._spherical_harmonics = _SphericalHarmonics(
            dimension=dim + 1, degrees=self._num_levels
        )

    def degree_to_num_eigenfunctions(self, degree: int) -> int:
        """
        Returns the number of eigenfunctions that span the first `degree` degrees.
        """
        n = 0
        for d in range(degree):
            n += num_harmonics(self.dim + 1, d)
        return n

    def num_eigenfunctions_to_degree(self, num_eigenfunctions: int) -> Tuple[int, int]:
        """
        Returns the minimum degree for which there are at least
        `num_eigenfunctions` in the collection.
        """
        n, d = 0, 0  # n: number of harmonics, d: degree (or level)
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
        return B.concat(*values, axis=-1)  # [N, N2, L]

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

    @classmethod
    def from_levels(cls, dimension, num_levels):
        num_eigenfunctions = 0
        for i in range(num_levels):
            num_eigenfunctions += num_harmonics(dimension, i)

        return cls(dimension, num_eigenfunctions)


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

    @property
    def dimension(self) -> int:
        return self.dim

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        :param num: number of eigenlevels (number of eigenspaces, less than or
                    equal to the number of eigenfunctions).
        """
        return SphericalHarmonics(self.dim, num)

    def get_eigenvalues(self, num: int) -> B.Numeric:
        """
        First `num` eigenvalues of the Laplace-Beltrami operator

        :return: [num, 1] array containing the eigenvalues
        """
        eigenfunctions = SphericalHarmonics(self.dim, num)
        eigenvalues = np.array(
            [
                level.eigenvalue()
                for level in eigenfunctions._spherical_harmonics.harmonic_levels
            ]
        )
        return B.reshape(eigenvalues, -1, 1)  # [num, 1]

    def get_repeated_eigenvalues(self, num: int) -> B.Numeric:
        """First `num` eigenvalues of the Laplace-Beltrami operator,
        repeated according to their multiplicity.

        :return: [M, 1] array containing the eigenvalues
        """
        eigenfunctions = SphericalHarmonics(self.dim, num)
        eigenvalues_per_level = np.array(
            [
                level.eigenvalue()
                for level in eigenfunctions._spherical_harmonics.harmonic_levels
            ]
        )
        eigenvalues = chain(
            eigenvalues_per_level,
            eigenfunctions.num_eigenfunctions_per_level,
        )  # [M,]
        return B.reshape(eigenvalues, -1, 1)  # [M, 1]

    def ehess2rhess(self, x, egrad, ehess, direction):
        """
        Riemannian Hessian along a given direction computed from the Euclidean Hessian

        :return: [dim] array containing Hess_f(x)[direction]

        References:

        [1] P.-A. Absil, R. Mahony, R. Sepulchre.
            Optimization algorithms on matrix manifolds. Princeton University Press 2007.
        """
        normal_gradient = egrad - self.to_tangent(egrad, x)
        return (
            self.to_tangent(ehess, x)
            - self.metric.inner_product(x, normal_gradient, x) * direction
        )

    def random(self, key, number):
        """
        Random points on the sphere.

        Always returns [N, D+1] float64 array of the `key`'s backend.
        """
        key, random_points = B.random.randn(
            key, dtype_double(key), number, self.dimension + 1
        )  # (N, D+1)
        random_points /= B.sqrt(
            B.sum(random_points**2, axis=1, squeeze=False)
        )  # (N, D+1)
        return key, random_points
