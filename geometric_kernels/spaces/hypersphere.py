"""
This module provides the :class:`Hypersphere` space and the respective
:class:`~.eigenfunctions.Eigenfunctions` subclass :class:`SphericalHarmonics`.
"""

import geomstats as gs
import lab as B
import numpy as np
from beartype.typing import List, Optional
from spherical_harmonics import SphericalHarmonics as _SphericalHarmonics
from spherical_harmonics.fundamental_set import num_harmonics

from geometric_kernels.lab_extras import dtype_double
from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import (
    Eigenfunctions,
    EigenfunctionsWithAdditionTheorem,
)
from geometric_kernels.utils.utils import chain


class SphericalHarmonics(EigenfunctionsWithAdditionTheorem):
    r"""
    Eigenfunctions of the Laplace-Beltrami operator on the hypersphere
    correspond to the spherical harmonics.

    Levels are the whole eigenspaces.

    :param dim:
        Dimension of the hypersphere.

        E.g. dim = 2 means the standard 2-dimensional sphere in $\mathbb{R}^3$.
        We only support dim >= 2. For dim = 1, use :class:`~.spaces.Circle`.

    :param num_levels:
        Specifies the number of levels of the spherical harmonics.
    """

    def __init__(self, dim: int, num_levels: int) -> None:
        self.dim = dim
        self._num_levels = num_levels
        self._num_eigenfunctions: Optional[int] = None  # To be computed when needed.
        self._spherical_harmonics = _SphericalHarmonics(
            dimension=dim + 1,
            degrees=self._num_levels,
            allow_uncomputed_levels=True,
        )

    @property
    def num_computed_levels(self) -> int:
        num = 0
        for level in self._spherical_harmonics.harmonic_levels:
            if level.is_level_computed:
                num += 1
            else:
                break
        return num

    def __call__(self, X: B.Numeric, **kwargs) -> B.Numeric:
        return self._spherical_harmonics(X)

    def _addition_theorem(
        self, X: B.Numeric, X2: Optional[B.Numeric] = None, **kwargs
    ) -> B.Numeric:
        r"""
        Returns the result of applying the addition theorem to sum over all
        the outer products of eigenfunctions within a level, for each level.

        The case of the hypersphere is explicitly considered on
        doc:`this documentation page </theory/addition_theorem>`.

        :param X:
            The first of the two batches of points to evaluate the phi
            product at. An array of shape [N, <axis>], where N is the
            number of points and <axis> is the shape of the arrays that
            represent the points in a given space.
        :param X2:
            The second of the two batches of points to evaluate the phi
            product at. An array of shape [N2, <axis>], where N2 is the
            number of points and <axis> is the shape of the arrays that
            represent the points in a given space.

            Defaults to None, in which case X is used for X2.
        :param ``**kwargs``:
            Any additional parameters.

        :return:
            An array of shape [N, N2, L].
        """
        values = [
            level.addition(X, X2)[..., None]  # [N, N2, 1]
            for level in self._spherical_harmonics.harmonic_levels
        ]
        return B.concat(*values, axis=-1)  # [N, N2, L]

    def _addition_theorem_diag(self, X: B.Numeric, **kwargs) -> B.Numeric:
        """
        These are certain easy to compute constants.
        """
        values = [
            level.addition_at_1(X)  # [N, 1]
            for level in self._spherical_harmonics.harmonic_levels
        ]
        return B.concat(*values, axis=1)  # [N, L]

    @property
    def num_eigenfunctions(self) -> int:
        if self._num_eigenfunctions is None:
            self._num_eigenfunctions = sum(self.num_eigenfunctions_per_level)
        return self._num_eigenfunctions

    @property
    def num_levels(self) -> int:
        return self._num_levels

    @property
    def num_eigenfunctions_per_level(self) -> List[int]:
        return [num_harmonics(self.dim + 1, level) for level in range(self.num_levels)]


class Hypersphere(DiscreteSpectrumSpace, gs.geometry.hypersphere.Hypersphere):
    r"""
    The GeometricKernels space representing the d-dimensional hypersphere
    $\mathbb{S}_d$ embedded in the (d+1)-dimensional Euclidean space.

    The elements of this space are represented by (d+1)-dimensional vectors
    of unit norm.

    Levels are the whole eigenspaces.

    .. note::
        We only support d >= 2. For d = 1, use :class:`~.spaces.Circle`.

    .. note::
        A tutorial on how to use this space is available in the
        :doc:`Hypersphere.ipynb </examples/Hypersphere>` notebook.

    :param dim:
        Dimension of the hypersphere $\mathbb{S}_d$.
        Should satisfy dim >= 2. For dim = 1, use :class:`~.spaces.Circle`.

    .. admonition:: Citation

        If you use this GeometricKernels space in your research, please consider
        citing :cite:t:`borovitskiy2020`.
    """

    def __init__(self, dim: int):
        if dim < 2:
            raise ValueError("Only dim >= 2 is supported. For dim = 1, use Circle.")
        super().__init__(dim=dim)
        self.dim = dim

    def __str__(self):
        return f"Hypersphere({self.dim})"

    @property
    def dimension(self) -> int:
        """
        Returns d, the `dim` parameter that was passed down to `__init__`.
        """
        return self.dim

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        Returns the :class:`~.SphericalHarmonics` object with `num` levels.

        :param num:
            Number of levels.
        """
        return SphericalHarmonics(self.dim, num)

    def get_eigenvalues(self, num: int) -> B.Numeric:
        eigenfunctions = SphericalHarmonics(self.dim, num)
        eigenvalues = np.array(
            [
                level.eigenvalue()
                for level in eigenfunctions._spherical_harmonics.harmonic_levels
            ]
        )
        return B.reshape(eigenvalues, -1, 1)  # [num, 1]

    def get_repeated_eigenvalues(self, num: int) -> B.Numeric:
        eigenfunctions = SphericalHarmonics(self.dim, num)
        eigenvalues_per_level = self.get_eigenvalues(num)

        eigenvalues = chain(
            B.squeeze(eigenvalues_per_level),
            eigenfunctions.num_eigenfunctions_per_level,
        )  # [J,]
        return B.reshape(eigenvalues, -1, 1)  # [J, 1]

    def ehess2rhess(
        self,
        x: B.NPNumeric,
        egrad: B.NPNumeric,
        ehess: B.NPNumeric,
        direction: B.NPNumeric,
    ) -> B.NPNumeric:
        """
        Riemannian Hessian along a given direction from the Euclidean Hessian.

        Used to test that the heat kernel does indeed solve the heat equation.

        :param x:
            A point on the d-dimensional hypersphere.
        :param egrad:
            Euclidean gradient of a function defined in a neighborhood of the
            hypersphere, evaluated at the point `x`.
        :param ehess:
            Euclidean Hessian of a function defined in a neighborhood of the
            hypersphere, evaluated at the point `x`.
        :param direction:
            Direction to evaluate the Riemannian Hessian at. A tangent vector
            at `x`.

        :return:
            A [dim]-shaped array that contains Hess_f(x)[direction], the value
            of the Riemannian Hessian of the function evaluated at `x` along
            the `direction`.

        See :cite:t:`absil2008` for mathematical details.
        """
        normal_gradient = egrad - self.to_tangent(egrad, x)
        return (
            self.to_tangent(ehess, x)
            - self.metric.inner_product(x, normal_gradient, x) * direction
        )

    def random(self, key: B.RandomState, number: int) -> B.Numeric:
        """
        Sample uniformly random points on the hypersphere.

        Always returns [N, D+1] float64 array of the `key`'s backend.

        :param key:
            Either `np.random.RandomState`, `tf.random.Generator`,
            `torch.Generator` or `jax.tensor` (representing random state).
        :param number:
            Number N of samples to draw.

        :return:
            An array of `number` uniformly random samples on the space.
        """
        key, random_points = B.random.randn(
            key, dtype_double(key), number, self.dimension + 1
        )  # (N, d+1)
        random_points /= B.sqrt(
            B.sum(random_points**2, axis=1, squeeze=False)
        )  # (N, d+1)
        return key, random_points

    @property
    def element_shape(self):
        """
        :return:
            [d+1].
        """
        return [self.dimension + 1]

    @property
    def element_dtype(self):
        """
        :return:
            B.Float.
        """
        return B.Float
