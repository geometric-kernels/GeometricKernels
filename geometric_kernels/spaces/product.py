"""
This module provides the :class:`ProductDiscreteSpectrumSpace` space and the
representation of its spectrum, the :class:`ProductEigenfunctions` class.
"""

import itertools

import lab as B
import numpy as np
from beartype.typing import List, Optional

from geometric_kernels.lab_extras import from_numpy
from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import Eigenfunctions
from geometric_kernels.utils.utils import chain


def _find_lowest_sum_combinations(array, k):
    """for an [N, D] array, assumed to be sorted within columns, find k smallest sums of
    one element from each row, and return the index array corresponding to this. Will
    possibly cause problems if k<D (but unlikely).
    """
    N, D = array.shape

    index_array = B.stack(*[B.zeros(int, N) for i in range(k)], axis=0)
    # prebuild array for indexing the first index of the eigenvalue array
    first_index = B.linspace(0, N - 1, N).astype(int)
    i = 0

    # first eigenvalue is the sum of the first eigenvalues of the individual spaces
    curr_idx = B.zeros(int, N)[None, :]
    while i < k:
        # compute eigenvalues of the proposals
        sum_values = array[first_index, curr_idx].sum(axis=1)

        # Compute tied smallest new eigenvalues
        lowest_sum_index = int(sum_values.argmin())
        tied_sums = sum_values == sum_values[lowest_sum_index]
        tied_sums_indexes = B.linspace(0, len(tied_sums) - 1, len(tied_sums)).astype(
            int
        )[tied_sums]

        # Add new eigenvalues to indexing array
        for index in tied_sums_indexes:
            index_array[i, :] = curr_idx[index]
            i += 1
            if i >= k:
                break

        # create new proposal eigenindicies

        # keep unaccepted ones around
        old_indices = curr_idx[~tied_sums]
        # mutate just accepted ones by adding one to each eigenindex
        new_indices = curr_idx[tied_sums][..., None, :] + B.eye(int, curr_idx.shape[-1])
        new_indices = new_indices.reshape((-1, new_indices.shape[-1]))
        curr_idx = B.concat(old_indices, new_indices, axis=0)
        curr_idx = np.unique(
            B.to_numpy(curr_idx.reshape((-1, curr_idx.shape[-1]))),
            axis=0,
        )
        # Filter out already searched combinations. See accepted answer of https://stackoverflow.com/questions/40055835/removing-elements-from-an-array-that-are-in-another-array
        dims = np.maximum(curr_idx.max(0), index_array.max(0)) + 1
        curr_idx = curr_idx[
            ~np.in1d(
                np.ravel_multi_index(curr_idx.T, dims),
                np.ravel_multi_index(index_array.T, dims),
            )
        ]

    return index_array


def _total_multiplicities(eigenindices, nums_per_level):
    """
    Given a collection of eigenindices [M, S],
    compute the total multiplicities of
    the corresponding eigenvalues.

    eigidx: [M, S]
    nums_per_level [S, L]
    """
    totals = []

    per_level = from_numpy(eigenindices, nums_per_level)
    num_sub_spaces = np.shape(eigenindices)[1]
    for index in eigenindices:
        multiplicities = per_level[B.range(num_sub_spaces), index]
        total = B.prod(multiplicities)
        totals.append(total)

        # totals = B.stack(*eigenindices, axis=0)
    return totals


def _num_per_level_to_mapping(num_per_level):
    mapping = []
    i = 0
    for num in num_per_level:
        mapping.append([i + j for j in range(num)])
        i += num
    return mapping


def _per_level_to_separate(eigenindices, nums_per_level):
    """
    Given `eigenindices` which map product space's eigenfunction index to
    the indices of subspaces' eigenlevels,
    convert them to a mapping of product space's eigenfunction index to
    the indices of subspaces' individual eigenfunctions via
    `nums_per_level`, which gives number of eigenfunctions per level for each subspace.

    :return: [M, S]
        `M` is the total number of eigenfunctions, `S` is the number of subspaces.
    """
    separate = [_num_per_level_to_mapping(npl) for npl in nums_per_level]
    # S lists of length L

    total_eigenindices = []
    for index in eigenindices:
        separates = [separate[s][level] for s, level in enumerate(index)]
        # S lists, each with separate indices
        new_indices = list(itertools.product(*separates))
        total_eigenindices += new_indices

    out = from_numpy(eigenindices, np.array(total_eigenindices))
    return out


class ProductEigenfunctions(Eigenfunctions):
    def __init__(
        self,
        dimensions: List[int],
        eigenindicies: B.Numeric,
        *eigenfunctions: Eigenfunctions,
        dimension_indices: B.Numeric = None,
    ):
        """
        Wrapper class for handling eigenfunctions on product spaces

        :param dimensions: the dimensions of the spaces being producted together

        :param eigenindicies: an array mapping i'th eigenfunction of the product
                              space to the index of the eigenlevels of the subspaces

        :param eigenfunctions: the eigenfunctions

        """
        if dimension_indices is None:
            self.dimension_indices = []
            i = 0
            inds = B.linspace(0, sum(dimensions) - 1, sum(dimensions)).astype(int)
            for dim in dimensions:
                self.dimension_indices.append(inds[i : i + dim])
                i += dim
        self.eigenindicies = eigenindicies
        self.eigenfunctions = eigenfunctions

        self.nums_per_level = [
            eigenfunction.num_eigenfunctions_per_level
            for eigenfunction in self.eigenfunctions
        ]  # [S, L]

        self._separate_eigenindices = _per_level_to_separate(
            self.eigenindicies, self.nums_per_level
        )

        assert self.eigenindicies.shape[-1] == len(self.eigenfunctions)

    def __call__(self, X: B.Numeric, **parameters) -> B.Numeric:
        Xs = [B.take(X, inds, axis=-1) for inds in self.dimension_indices]

        eigenfunctions = B.stack(
            *[
                eigenfunction(X, **parameters)  # [N, M]
                for eigenfunction, X in zip(self.eigenfunctions, Xs)
            ],
            axis=-1,
        )  # [N, M, S]

        # eigenindices shape [M, S]

        return eigenfunctions[
            :,
            self._separate_eigenindices,
            B.range(self.eigenindicies.shape[1]),
        ].prod(
            axis=-1
        )  # [N, M, S] --> [N, M]

    @property
    def num_eigenfunctions(self) -> int:
        """
        Return the total number of eigenfunctions.
        """
        return self._separate_eigenindices.shape[0]

    @property
    def num_levels(self) -> int:
        """
        Return number of "levels".
        """
        return self.eigenindicies.shape[0]

    def phi_product(
        self, X: B.Numeric, X2: Optional[B.Numeric] = None, **kwargs
    ) -> B.Numeric:
        Xs = [B.take(X, inds, axis=-1) for inds in self.dimension_indices]
        Xs2 = [B.take(X2, inds, axis=-1) for inds in self.dimension_indices]

        phis = B.stack(
            *[
                eigenfunction.phi_product(X1, X2, **kwargs)
                for eigenfunction, X1, X2 in zip(self.eigenfunctions, Xs, Xs2)
            ],
            axis=-1,
        )  # [N, M, L, S]

        prod_phis = phis[
            :,
            :,
            self.eigenindicies,
            B.range(self.eigenindicies.shape[1]),
        ].prod(
            axis=-1
        )  # [N, M, L, S] -> [N, M, L]

        return prod_phis

    def phi_product_diag(self, X: B.Numeric, **kwargs):
        Xs = [B.take(X, inds, axis=-1) for inds in self.dimension_indices]

        phis = B.stack(
            *[
                eigenfunction.phi_product_diag(X1, **kwargs)
                for eigenfunction, X1 in zip(self.eigenfunctions, Xs)
            ],
            axis=-1,
        )  # [N, L, S]

        prod_phis = phis[
            :,
            self.eigenindicies,
            B.range(self.eigenindicies.shape[1]),
        ].prod(
            axis=-1
        )  # [N, L, S] -> [N, L]

        return prod_phis

    @property
    def num_eigenfunctions_per_level(self) -> List[int]:
        return _total_multiplicities(self.eigenindicies, self.nums_per_level)


class ProductDiscreteSpectrumSpace(DiscreteSpectrumSpace):
    def __init__(self, *spaces: DiscreteSpectrumSpace, num_levels: int = 25):
        r"""Implementation of products of discrete spectrum spaces.
        Assumes the spaces are compact manifolds and that the eigenfunctions are the
        eigenfunctions of the Laplace-Beltrami operator.

        Denote a product space :math:`\mathcal{S} = \mathcal{S}_1 \times \ldots \mathcal{S}_S`.

        Eigenvalues on the product space are sums of the factors' eigenvalues:

        .. math ::
            \lambda_{l_1, \ldots, l_S} = \lambda^{1}_{l_1} + \ldots + \lambda^{S}_{l_S}

        Each factor's eigenvalue corresponds to the factor's eigenfunctions (perhaps multiple eigenfunctions):

        .. math ::
            \lambda^{s}_{l_s} \leftrightarrow (f^{s}_{l_s, 1}, \ldots, f^{s}_{l_s, J_{l_s}})

        This is referred to as a level.
        Product-space eigenfunctions are products of factors' eigenfunctions within each level.

        Whenever factors' eigenfunctions are grouped in a level, this induces the product-space
        eigenfunction to be group in a level. Thus, we operate on levels.

        The product-space levels can't in general be analytically ordered, and
        so they must be precomputed.

        :param spaces: The spaces to product together (each must inherit from DiscreteSpectrumSpace)
        :param num_levels: (optional)
            number of levels to pre-compute for this product space.
        """
        for space in spaces:
            assert isinstance(
                space, DiscreteSpectrumSpace
            ), "One of the spaces is not an instance of DiscreteSpectrumSpace."

        self.sub_spaces = spaces
        self.num_eigen = num_levels

        # perform an breadth-first search for the smallest eigenvalues,
        # assuming that the eigenvalues come sorted,the next biggest eigenvalue
        # can be found by taking a one-index step in any direction from the current
        # edge of the searchspace

        # prefetch the eigenvalues of the subspaces
        sub_space_eigenvalues = B.stack(
            *[space.get_eigenvalues(self.num_eigen)[:, 0] for space in self.sub_spaces],
            axis=0,
        )  # [M, S]

        self.sub_space_eigenindices = _find_lowest_sum_combinations(
            sub_space_eigenvalues, self.num_eigen
        )
        self.sub_space_eigenvalues = sub_space_eigenvalues

        self._eigenvalues = sub_space_eigenvalues[
            B.range(len(self.sub_spaces)),
            self.sub_space_eigenindices[: self.num_eigen, :],
        ].sum(axis=1)

    @property
    def dimension(self) -> int:
        return sum([space.dimension for space in self.sub_spaces])

    def random(self, key, number):
        random_points = []
        for factor in self.sub_spaces:
            key, factor_random_points = factor.random(key, number)
            random_points.append(factor_random_points)

        return key, B.concat(*random_points, axis=1)

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        assert num <= self.num_eigen

        max_eigenvalue = self.sub_space_eigenindices[:num, :].max() + 1

        sub_space_eigenfunctions = [
            space.get_eigenfunctions(max_eigenvalue) for space in self.sub_spaces
        ]

        return ProductEigenfunctions(
            [space.dimension for space in self.sub_spaces],
            self.sub_space_eigenindices,
            *sub_space_eigenfunctions,
        )

    def get_eigenvalues(self, num: int) -> B.Numeric:
        assert num <= self.num_eigen

        return self._eigenvalues[:num, None]

    def get_repeated_eigenvalues(self, num: int) -> B.Numeric:
        assert num <= self.num_eigen

        eigenfunctions = self.get_eigenfunctions(num)
        eigenvalues = self._eigenvalues[:num]
        multiplicities = eigenfunctions.num_eigenfunctions_per_level

        repeated_eigenvalues = chain(eigenvalues, multiplicities)
        return B.reshape(repeated_eigenvalues, -1, 1)  # [M, 1]
