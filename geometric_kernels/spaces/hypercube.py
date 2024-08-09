"""
This module provides the :class:`Hypercube` space and the respective
:class:`~.eigenfunctions.Eigenfunctions` subclass :class:`WalshFunctions`.
"""

from itertools import combinations
from math import comb

import lab as B
import numpy as np
from beartype.typing import List, Optional

from geometric_kernels.lab_extras import dtype_double, float_like
from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import (
    Eigenfunctions,
    EigenfunctionsWithAdditionTheorem,
)
from geometric_kernels.utils.special_functions import (
    kravchuk_normalized,
    walsh_function,
)
from geometric_kernels.utils.utils import chain, hamming_distance


class WalshFunctions(EigenfunctionsWithAdditionTheorem):
    r"""
    Eigenfunctions of graph Laplacian on the hypercube $C = \{0, 1\}^d$ are
    Walsh functions $w_T: C \to \{-1, 1\}$ given by

    .. math:: w_T(x_1, .., x_d) = (-1)^{\sum_{i \in T} x_i},

    enumerated by all possible subsets $T$ of the set $\{1, .., d\}$.

    .. note::
        Since the degree matrix is a constant multiple of the identity, all
        types of the graph Laplacian coincide on the hypercube up to a constant.

    Levels are the whole eigenspaces, comprising all Walsh functions $w_T$ with
    the same cardinality of $T$. The addition theorem for these is based on
    dynamically precomputed basis functions equivalent to certain
    discretizations of the Kravchuk polynomials.

    :param dim:
        Dimension $d$ of the hypercube.

    :param num_levels:
        Specifies the number of levels of the spherical harmonics.

    :todo:
        Implement explicit weighted_outerproduct and weighted_outerproduct_diag
        which compute the product of weights and binom(d, j) in log scale (for
        numerical stability).
    """

    def __init__(self, dim: int, num_levels: int) -> None:
        assert num_levels <= dim + 1, "The number of levels should be at most dim+1."
        self.dim = dim
        self._num_levels = num_levels
        self._num_eigenfunctions: Optional[int] = None  # To be computed when needed.

    def __call__(self, X: B.Bool, **kwargs) -> B.Float:
        return B.stack(
            *[
                walsh_function(self.dim, list(cur_combination), X)
                for level in range(self.num_levels)
                for cur_combination in combinations(range(self.dim), level)
            ],
            axis=1,
        )

    def _addition_theorem(
        self, X: B.Numeric, X2: Optional[B.Numeric] = None, **kwargs
    ) -> B.Numeric:

        if X2 is None:
            X2 = X

        hamming_distances = hamming_distance(X, X2)

        values = [
            comb(self.dim, level)
            * kravchuk_normalized(self.dim, level, hamming_distances)[
                ..., None
            ]  # [N, N2, 1]
            for level in range(self.num_levels)
        ]

        return B.concat(*values, axis=-1)  # [N, N2, L]

    def _addition_theorem_diag(self, X: B.Numeric, **kwargs) -> B.Numeric:
        """
        These are certain easy to compute constants.
        """
        values = [
            comb(self.dim, level) * B.ones(float_like(X), *X.shape[:-1], 1)  # [N, 1]
            for level in range(self.num_levels)
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
        return [comb(self.dim, level) for level in range(self.num_levels)]


class Hypercube(DiscreteSpectrumSpace):
    r"""
    The GeometricKernels space representing the d-dimensional hypercube graph
    $C = \{0, 1\}^d$, the combinatorial space of binary vectors of length $d$.

    The elements of this space are represented by d-dimensional boolean vectors.

    Levels are the whole eigenspaces.

    .. note::
        A tutorial on how to use this space is available in the
        :doc:`Hypersphere.ipynb </examples/Hypercube>` notebook.

    :param dim:
        Dimension $d$ of the hypercube $C = \{0, 1\}^d$, a positive integer.

    .. admonition:: Citation

        If you use this GeometricKernels space in your research, please consider
        citing :cite:t:`borovitskiy2023`.
    """

    def __init__(self, dim: int):
        if dim < 1:
            raise ValueError("dim must be a positive integer.")
        self.dim = dim

    @property
    def dimension(self) -> int:
        """
        Returns d, the `dim` parameter that was passed down to `__init__`.

        .. note:
            Although this is a graph, and graphs are generally treated as
            0-dimensional throughout GeometricKernels, we make an exception for
            the hypercube. This is because it helps maintain good behavior of
            Matérn kernels with the usual values of the smoothness parameter
            nu, i.e. nu = 1/2, nu=3/2, nu=5/2.
        """
        return self.dim

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        Returns the :class:`~.SphericalHarmonics` object with `num` levels.

        :param num:
            Number of levels.
        """
        return WalshFunctions(self.dim, num)

    def get_eigenvalues(self, num: int) -> B.Numeric:
        eigenvalues = np.array(
            [
                2
                * level
                / self.dim  # we assume normalized Laplacian (for numerical stability)
                for level in range(num)
            ]
        )
        return B.reshape(eigenvalues, -1, 1)  # [num, 1]

    def get_repeated_eigenvalues(self, num: int) -> B.Numeric:
        eigenvalues_per_level = self.get_eigenvalues(num)

        eigenfunctions = WalshFunctions(self.dim, num)
        eigenvalues = chain(
            B.squeeze(eigenvalues_per_level),
            eigenfunctions.num_eigenfunctions_per_level,
        )  # [J,]
        return B.reshape(eigenvalues, -1, 1)  # [J, 1]

    def random(self, key: B.RandomState, number: int) -> B.Numeric:
        r"""
        Sample uniformly random points on the hypercube $C = \{0, 1\}^d$.

        Always returns [N, D] boolean array of the `key`'s backend.

        :param key:
            Either `np.random.RandomState`, `tf.random.Generator`,
            `torch.Generator` or `jax.tensor` (representing random state).
        :param number:
            Number N of samples to draw.

        :return:
            An array of `number` uniformly random samples on the space.
        """
        key, random_points = B.random.rand(
            key, dtype_double(key), number, self.dimension
        )

        random_points = random_points < 0.5

        return key, random_points

    @property
    def element_shape(self):
        """
        :return:
            [d].
        """
        return [self.dimension]
