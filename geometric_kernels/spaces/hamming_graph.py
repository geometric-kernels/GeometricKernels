"""
This module provides the :class:`HammingGraph` space and the respective
:class:`~.eigenfunctions.Eigenfunctions` subclass :class:`VilenkinFunctions`.
"""

from math import comb

import lab as B
import numpy as np
from beartype.typing import List, Optional

from geometric_kernels.lab_extras import dtype_integer, float_like
from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import (
    Eigenfunctions,
    EigenfunctionsWithAdditionTheorem,
)
from geometric_kernels.utils.special_functions import generalized_kravchuk_normalized
from geometric_kernels.utils.utils import chain, hamming_distance, log_binomial


class VilenkinFunctions(EigenfunctionsWithAdditionTheorem):
    r"""
    Eigenfunctions of the graph Laplacian on the q-ary Hamming graph $H(d,q)$, whose
    nodes are indexed by categorical vectors in $\{0, 1, ..., q-1\}^d$.

    These eigenfunctions are the Vilenkin functions (also called Vilenkin-Chrestenson
    functions), which generalize the binary Walsh functions to q-ary alphabets. They
    map vertices to complex values via products of characters on cyclic groups.

    For the special case $q = 2$, the Vilenkin functions reduce to the Walsh functions
    on the binary hypercube $\{0, 1\}^d$.

    .. note::
        The Vilenkin functions can be indexed by "character patterns" - choices of
        coordinates and non-identity characters at those coordinates. Each eigenspace
        (level) $j$ has dimension $\binom{d}{j}(q-1)^j$, corresponding to choosing
        $j$ coordinates and assigning $(q-1)$ possible non-identity characters to each.

    Levels are the whole eigenspaces, comprising all Vilenkin functions with the
    same number of coordinates having non-identity characters. The addition theorem
    for these is based on generalized Kravchuk polynomials, i.e. discrete orthogonal
    polynomials on the q-ary Hamming scheme.

    :param dim:
        Dimension $d$ of the q-ary Hamming graph $H(d,q)$.

    :param n_cat:
        Number of categories $q \geq 2$ in the q-ary alphabet $\{0, 1, ..., q-1\}$.

    :param num_levels:
        Specifies the number of levels (eigenspaces) of the Vilenkin functions to use.
    """

    def __init__(self, dim: int, n_cat: int, num_levels: int) -> None:
        if num_levels > dim + 1:
            raise ValueError("The number of levels should be at most `dim`+1.")
        self.dim = dim
        self.n_cat = n_cat
        self._num_levels = num_levels
        self._num_eigenfunctions: Optional[int] = None  # To be computed when needed.

        if n_cat < 2:
            raise ValueError("n_cat must be at least 2.")

    def __call__(self, X: B.Int, **kwargs) -> B.Float:
        raise NotImplementedError

    def _addition_theorem(
        self, X: B.Numeric, X2: Optional[B.Numeric] = None, **kwargs
    ) -> B.Numeric:

        if X2 is None:
            X2 = X

        hamming_distances = hamming_distance(X, X2)

        values = []

        kravchuk_normalized_j_minus_1, kravchuk_normalized_j_minus_2 = None, None
        for level in range(self.num_levels):
            cur_kravchuk_normalized = generalized_kravchuk_normalized(
                self.dim,
                level,
                hamming_distances,
                self.n_cat,
                kravchuk_normalized_j_minus_1,
                kravchuk_normalized_j_minus_2,
            )  # [N, N2]
            kravchuk_normalized_j_minus_2 = kravchuk_normalized_j_minus_1
            kravchuk_normalized_j_minus_1 = cur_kravchuk_normalized

            values.append(
                comb(self.dim, level)
                * (self.n_cat - 1) ** level
                * cur_kravchuk_normalized[..., None]
            )  # [N, N2, 1]

        return B.concat(*values, axis=-1)  # [N, N2, L]

    def _addition_theorem_diag(self, X: B.Numeric, **kwargs) -> B.Numeric:
        """
        These are certain easy to compute constants.
        """
        values = [
            comb(self.dim, level)
            * (self.n_cat - 1) ** level
            * B.ones(float_like(X), *X.shape[:-1], 1)  # [N, 1]
            for level in range(self.num_levels)
        ]
        return B.concat(*values, axis=1)  # [N, L]

    def weighted_outerproduct(
        self,
        weights: B.Numeric,
        X: B.Numeric,
        X2: Optional[B.Numeric] = None,  # type: ignore
        **kwargs,
    ) -> B.Numeric:
        if X2 is None:
            X2 = X

        hamming_distances = hamming_distance(X, X2)

        result = B.zeros(B.dtype(weights), X.shape[0], X2.shape[0])  # [N, N2]
        kravchuk_normalized_j_minus_1, kravchuk_normalized_j_minus_2 = None, None
        for level in range(self.num_levels):
            cur_kravchuk_normalized = generalized_kravchuk_normalized(
                self.dim,
                level,
                hamming_distances,
                self.n_cat,
                kravchuk_normalized_j_minus_1,
                kravchuk_normalized_j_minus_2,
            )
            kravchuk_normalized_j_minus_2 = kravchuk_normalized_j_minus_1
            kravchuk_normalized_j_minus_1 = cur_kravchuk_normalized

            # Instead of multiplying weights by binomial coefficients, we sum their
            # logs and then exponentiate the result for numerical stability.
            # Furthermore, we save the computed Kravchuk polynomials for next iterations.
            result += (
                B.exp(
                    B.log(weights[level])
                    + log_binomial(self.dim, level)
                    + level * B.log(self.n_cat - 1)
                )
                * cur_kravchuk_normalized
            )

        return result  # [N, N2]

    def weighted_outerproduct_diag(
        self, weights: B.Numeric, X: B.Numeric, **kwargs
    ) -> B.Numeric:

        # Instead of multiplying weights by binomial coefficients, we sum their
        # logs and then exponentiate the result for numerical stability.
        result = sum(
            B.exp(
                B.log(weights[level])
                + log_binomial(self.dim, level)
                + level * B.log(self.n_cat - 1)
            )
            * B.ones(float_like(X), *X.shape[:-1], 1)
            for level in range(self.num_levels)
        )  # [N, 1]

        return B.reshape(result, *result.shape[:-1])  # [N,]

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
        return [
            comb(self.dim, level) * (self.n_cat - 1) ** level
            for level in range(self.num_levels)
        ]


class HammingGraph(DiscreteSpectrumSpace):
    r"""
    The GeometricKernels space representing the q-ary Hamming graph
    $H(d,q) = \{0, 1, ..., q-1\}^d$, the combinatorial space of categorical
    vectors (with $q$ categories) of length $d$.

    The elements of this space are represented by d-dimensional categorical vectors
    (with $q$ categories) taking integer values in $\{0, 1, ..., q-1\}$.

    Levels are the whole eigenspaces.

    .. note::
        If you need a kernel operating on categorical vectors where $q$ varies
        between dimensions, you can use `HammingGraph` in conjunction with
        :class:`ProductGeometricKernel` or :class:`ProductDiscreteSpectrumSpace`.

    .. note::
        For the special case $q = 2$, this reduces to the binary hypercube graph,
        also available as :class:`HypercubeGraph`.

    .. note::
        A tutorial on how to use this space is available in the
        :doc:`HammingGraph.ipynb </examples/HammingGraph>` notebook.

    .. note::
        Since the degree matrix is a constant multiple of the identity, all
        types of the graph Laplacian coincide on the Hamming graph up to a
        constant, we choose the normalized Laplacian for numerical stability.

    :param dim:
        Dimension $d$ of the Hamming graph $H(d,q)$, a positive integer.

    :param n_cat:
        Number of categories $q$ of the Hamming graph $H(d,q)$, a positive
        integer $q \geq 2$.

    .. admonition:: Citation

        If you use this GeometricKernels space in your research, please consider
        citing :cite:t:`borovitskiy2023` and :cite:t:`doumont2025`.
    """

    def __init__(self, dim: int, n_cat: int):
        if dim < 1:
            raise ValueError("dim must be a positive integer.")
        if n_cat < 1:
            raise ValueError("n_cat must be a positive integer.")
        self.dim = dim
        self.n_cat = n_cat

    def __str__(self):
        return f"HammingGraph({self.dim},{self.n_cat})"

    @property
    def dimension(self) -> int:
        """
        Returns d, the `dim` parameter that was passed down to `__init__`.

        .. note:
            Although this is a graph, and graphs are generally treated as
            0-dimensional throughout GeometricKernels, we make an exception for
            HammingGraph. This is because it helps maintain good behavior of
            Matérn kernels with the usual values of the smoothness parameter
            nu, i.e. nu = 1/2, nu = 3/2, nu = 5/2.
        """
        return self.dim

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        Returns the :class:`~.VilenkinFunctions` object with `num` levels.

        :param num:
            Number of levels.
        """
        return VilenkinFunctions(self.dim, self.n_cat, num)

    def get_eigenvalues(self, num: int) -> B.Numeric:
        eigenvalues = np.array(
            [
                (self.n_cat * level)
                / (
                    self.dim * (self.n_cat - 1)
                )  # we assume normalized Laplacian (for numerical stability)
                for level in range(num)
            ]
        )
        return B.reshape(eigenvalues, -1, 1)  # [num, 1]

    def get_repeated_eigenvalues(self, num: int) -> B.Numeric:
        eigenvalues_per_level = self.get_eigenvalues(num)

        eigenfunctions = VilenkinFunctions(self.dim, self.n_cat, num)
        eigenvalues = chain(
            B.squeeze(eigenvalues_per_level),
            eigenfunctions.num_eigenfunctions_per_level,
        )  # [J,]
        return B.reshape(eigenvalues, -1, 1)  # [J, 1]

    def random(self, key: B.RandomState, number: int) -> B.Numeric:
        r"""
        Sample uniformly random points on the Hamming graph $H(d,q)$.

        Always returns [N, D] integer array of the `key`'s backend with values
        in $\{0, 1, ..., q-1\}$.

        :param key:
            Either `np.random.RandomState`, `tf.random.Generator`,
            `torch.Generator` or `jax.tensor` (representing random state).
        :param number:
            Number N of samples to draw.

        :return:
            An array of `number` uniformly random samples on the space.
        """
        key, random_points = B.random.randint(
            key, dtype_integer(key), number, self.dimension, lower=0, upper=self.n_cat
        )

        return key, random_points

    @property
    def element_shape(self):
        """
        :return:
            [d].
        """
        return [self.dimension]

    @property
    def element_dtype(self):
        """
        :return:
            B.Int.
        """
        return B.Int
