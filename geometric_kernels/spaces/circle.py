"""
This module provides the :class:`Circle` space and the respective
:class:`~.eigenfunctions.Eigenfunctions` subclass :class:`SinCosEigenfunctions`.
"""

import lab as B
from beartype.typing import List, Optional

from geometric_kernels.lab_extras import dtype_double, from_numpy
from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import (
    Eigenfunctions,
    EigenfunctionsWithAdditionTheorem,
)
from geometric_kernels.utils.utils import chain


class SinCosEigenfunctions(EigenfunctionsWithAdditionTheorem):
    """
    Eigenfunctions of the Laplace-Beltrami operator on the circle correspond
    to the Fourier basis, i.e. sines and cosines.

    Levels are the whole eigenspaces. The zeroth eigenspace is
    one-dimensional, all the other eigenspaces are of dimension 2.

    :param num_levels:
        The number of levels.
    """

    def __init__(self, num_levels: int):
        assert num_levels >= 1

        self._num_eigenfunctions = num_levels * 2 - 1
        self._num_levels = num_levels

    def __call__(self, X: B.Numeric, **kwargs) -> B.Numeric:
        N = B.shape(X)[0]
        theta = X
        const = 2.0**0.5
        values = []
        for level in range(self.num_levels):
            if level == 0:
                values.append(B.ones(B.dtype(X), N, 1))
            else:
                freq = 1.0 * level
                values.append(const * B.cos(freq * theta))
                values.append(const * B.sin(freq * theta))

        return B.concat(*values, axis=1)[:, : self._num_eigenfunctions]  # [N, M]

    def _addition_theorem(
        self, X: B.Numeric, X2: Optional[B.Numeric] = None, **kwargs
    ) -> B.Numeric:
        r"""
        Returns the result of applying the addition theorem to sum over all
        the outer products of eigenfunctions within a level, for each level.

        .. math:: \sin(l \theta_1) \sin(l \theta_2) + \cos(l \theta_1) \cos(l \theta_2 = N_l \cos(l (\theta_1 - \theta_2)),

        where $N_l = 1$ for $l = 0$, else $N_l = 2$.

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
        theta1, theta2 = X, X2
        angle_between = theta1[:, None, :] - theta2[None, :, :]  # [N, N2, 1]
        freqs = B.range(B.dtype(X), self.num_levels)  # [L]
        values = B.cos(freqs[None, None, :] * angle_between)  # [N, N2, L]
        values = (
            B.cast(
                B.dtype(X),
                from_numpy(X, self.num_eigenfunctions_per_level)[None, None, :],
            )
            * values
        )
        return values  # [N, N2, L]

    def _addition_theorem_diag(self, X: B.Numeric, **kwargs) -> B.Numeric:
        """
        :return:
            Array `result`, such that result[n, l] = 1 if l = 0 or 2 otherwise.
        """
        N = X.shape[0]
        ones = B.ones(B.dtype(X), N, self.num_levels)  # [N, L]
        value = ones * B.cast(
            B.dtype(X), from_numpy(X, self.num_eigenfunctions_per_level)[None, :]
        )
        return value  # [N, L]

    @property
    def num_eigenfunctions(self) -> int:
        return self._num_eigenfunctions

    @property
    def num_levels(self) -> int:
        return self._num_levels

    @property
    def num_eigenfunctions_per_level(self) -> List[int]:
        """
        The number of eigenfunctions per level.

        :return:
            List `result`, such that result[l] = 1 if l = 0 or 2 otherwise.
        """
        return [1 if level == 0 else 2 for level in range(self.num_levels)]


class Circle(DiscreteSpectrumSpace):
    r"""
    The GeometricKernels space representing the standard unit circle, denoted
    by $\mathbb{S}_1$ (as the one-dimensional hypersphere) or $\mathbb{T}$ (as
    the one-dimensional torus).

    The elements of this space are represented by angles,
    scalars from $0$ to $2 \pi$.

    Levels are the whole eigenspaces. The zeroth eigenspace is
    one-dimensional, all the other eigenspaces are of dimension 2.

    .. note::
        The :doc:`example notebook on the torus </examples/Torus>` involves
        this space.

    .. admonition:: Citation

        If you use this GeometricKernels space in your research, please consider
        citing :cite:t:`borovitskiy2020`.
    """

    def __str__(self):
        return "Circle()"

    @property
    def dimension(self) -> int:
        """
        :return:
            1.
        """
        return 1

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        Returns the :class:`~.SinCosEigenfunctions` object with `num` levels.

        :param num:
            Number of levels.
        """
        return SinCosEigenfunctions(num)

    def get_eigenvalues(self, num: int) -> B.Numeric:
        eigenvalues = B.range(num) ** 2  # [num,]
        return B.reshape(eigenvalues, -1, 1)  # [num, 1]

    def get_repeated_eigenvalues(self, num: int) -> B.Numeric:
        eigenfunctions = self.get_eigenfunctions(num)
        eigenvalues_per_level = B.range(num) ** 2  # [num,]
        eigenvalues = chain(
            eigenvalues_per_level,
            eigenfunctions.num_eigenfunctions_per_level,
        )  # [M,]
        return B.reshape(eigenvalues, -1, 1)  # [M, 1]

    def random(self, key: B.RandomState, number: int):
        key, random_points = B.random.rand(key, dtype_double(key), number, 1)  # [N, 1]
        random_points *= 2 * B.pi
        return key, random_points

    @property
    def element_shape(self):
        """
        :return:
            [1].
        """
        return [1]

    @property
    def element_dtype(self):
        """
        :return:
            B.Float.
        """
        return B.Float
