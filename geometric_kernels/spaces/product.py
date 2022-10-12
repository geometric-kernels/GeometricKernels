"""
Implement product spaces
"""

import itertools
from typing import List

import lab as B
import numpy as np

from geometric_kernels.lab_extras import from_numpy
from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import Eigenfunctions
from geometric_kernels.utils.utils import chain


def find_lowest_sum_combinations(array, k):
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


def total_multiplicities(eigenindices, nums_per_level):
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


def num_per_level_to_mapping(num_per_level):
    mapping = []
    i = 0
    for num in num_per_level:
        mapping.append([i + j for j in range(num)])
        i += num
    return mapping


def per_level_to_separate(eigenindices, nums_per_level):
    """
    Given `eigenindices` which map product space's eigenfunction index to
    the indices of subspaces' eigenlevels,
    convert them to a mapping of product space's eigenfunction index to
    the indices of subspaces' individual eigenfunctions via
    `nums_per_level`, which gives number of eigenfunctions per level for each subspace.

    :return: [M, S]
        `M` is the total number of eigenfunctions, `S` is the number of subspaces.
    """
    separate = [num_per_level_to_mapping(npl) for npl in nums_per_level]
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

        Parameters
        ----------
        dimensions : List[int]
            The dimensions of the spaces being producted together
        eigenindicies : B.Numeric
            An array mapping i'th eigenfunction of the product space to
            the index of the eigenlevels of the subspaces
        eigenfunctions : Eigenfunctions

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
            eigenfunction.dim_of_eigenspaces for eigenfunction in self.eigenfunctions
        ]  # [S, L]

        self._separate_eigenindices = per_level_to_separate(
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

    def weighted_outerproduct(self, weights, X, X2=None, **parameters):
        if X2 is None:
            X2 = X
        Xs = [B.take(X, inds, axis=-1) for inds in self.dimension_indices]
        Xs2 = [B.take(X2, inds, axis=-1) for inds in self.dimension_indices]

        phis = B.stack(
            *[
                eigenfunction.phi_product(X1, X2, **parameters)
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

        # weights [L, 1]
        out = B.sum(B.flatten(weights) * prod_phis, axis=-1)  # [N, M, L] -> [N, M]

        return out

    def weighted_outerproduct_diag(
        self, weights: B.Numeric, X: B.Numeric, **parameters
    ) -> B.Numeric:
        Xs = [B.take(X, inds, axis=-1) for inds in self.dimension_indices]

        phis = B.stack(
            *[
                eigenfunction.phi_product_diag(X1, **parameters)
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

        out = B.sum(B.flatten(weights) * prod_phis, axis=-1)  # [N, L] -> [N,]
        return out

    @property
    def dim_of_eigenspaces(self):
        return total_multiplicities(self.eigenindicies, self.nums_per_level)


class ProductDiscreteSpectrumSpace(DiscreteSpectrumSpace):
    def __init__(self, *spaces: DiscreteSpectrumSpace, num_eigen: int = 100):
        r"""Implementation of products of discrete spectrum spaces.
        Assumes the spaces are compact manifolds and that the eigenfunctions are the
        eigenfunctions of the Laplace-Beltrami operator. On such a space the eigen(values/functions)
        on the product space associated with the multiindex alpha are given by
            lambda_alpha = \sum_i lambda_{i, alpha_i}
            phi_alpha = \prod_i phi_{i, alpha_i}
        where lambda_{i, j} is the j'th eigenvalue on the i'th manifold in the product
        and phi_{i, j} is the j'th eigenfunction on the i'th manifold in the product.

        The eigenfunctions of such manifolds can't in genreal be analytically ordered, and
        so they must be precomputed.

        Parameters
        ----------
        spaces : DiscreteSpecturmSpace
            The spaces to product together
        num_eigen : int, optional
            number of eigenvalues to use for this product space, by default 100
        """
        for space in spaces:
            assert isinstance(space, DiscreteSpectrumSpace)

        self.sub_spaces = spaces
        self.num_eigen = num_eigen

        # perform an breadth-first search for the smallest eigenvalues,
        # assuming that the eigenvalues come sorted,the next biggest eigenvalue
        # can be found by taking a one-index step in any direction from the current
        # edge of the searchspace

        # prefetch the eigenvalues of the subspaces
        sub_space_eigenvalues = B.stack(
            *[space.get_eigenvalues(self.num_eigen)[:, 0] for space in self.sub_spaces],
            axis=0,
        )  # [M, S]

        self.sub_space_eigenindices = find_lowest_sum_combinations(
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
        eigenvalues = self._eigenvalues[:num, None]
        multiplicities = eigenfunctions.dim_of_eigenspaces

        repeated_eigenvalues = chain(eigenvalues, multiplicities)
        return B.reshape(repeated_eigenvalues, -1, 1)  # [M, 1]
