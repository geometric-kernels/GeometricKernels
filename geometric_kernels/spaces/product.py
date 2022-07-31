"""
Implement product spaces
"""

from typing import List

import lab as B
import numpy as np

from .base import DiscreteSpectrumSpace

from geometric_kernels.spaces.eigenfunctions import (
    Eigenfunctions,
    EigenfunctionWithAdditionTheorem,
)


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


def indices_to_levels(index_array, level_mapping_array):
    """
    Takes and index array [N, D] containing indices and maps these indices to the levels they belong to,
    defined by the level_mapping_array, of size [D, levels, level_size].
    """
    # TODO: add some check on complete levels?
    N, D = index_array.shape
    level_array = index_array.copy()
    for d in range(D):
        max_index = index_array[:, d].max()
        for i in range(len(level_mapping_array[d])):
            for index in level_mapping_array[d][i]:
                if index <= max_index:
                    ids = level_array[:, d] == index
                    level_array[ids, d] = i

    _, unique_indexes = np.unique(level_array, axis=0, return_index=True)

    return level_array[np.sort(unique_indexes)]


def levels_to_indices(level_array, level_mapping_array):
    N, D = level_array.shape
    index_array = np.split(level_array, level_array.shape[0], axis=0)
    index_array = [i[0] for i in index_array]

    for d in range(D):
        new_index_array = []
        for index in index_array:
            for new_idx in level_mapping_array[d][index[d]]:
                new_index = index.copy()
                new_index[d] = new_idx
                new_index_array.append(new_index)

        index_array = new_index_array

    return np.stack(new_index_array, axis=0)


def num_per_level_to_mapping(num_per_level):
    mapping = []
    i = 0
    for num in num_per_level:
        mapping.append([i + j for j in range(num)])
        i += num


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
            the index of the eigenfunctions of the subspaces
        eigenfunctions : Eigenfunctions

        """
        if dimension_indices == None:
            self.dimension_indices = []
            i = 0
            inds = B.linspace(0, sum(dimensions) - 1, sum(dimensions)).astype(int)
            for dim in dimensions:
                self.dimension_indices.append(inds[i : i + dim])
                i += dim
        self.eigenindicies = eigenindicies
        self.eigenfunctions = eigenfunctions

        assert self.eigenindicies.shape[-1] == len(self.eigenfunctions)

    def __call__(self, X: B.Numeric, **parameters) -> B.Numeric:
        Xs = [B.take(X, inds, axis=-1) for inds in self.dimension_indices]

        eigenfunctions = B.stack(
            *[
                eigenfunction(X, **parameters)
                for eigenfunction, X in zip(self.eigenfunctions, Xs)
            ],
            axis=-1,
        )

        return eigenfunctions[
            :,
            self.eigenindicies,
            B.linspace(
                0, self.eigenindicies.shape[1] - 1, self.eigenindicies.shape[1]
            ).astype(int),
        ].prod(axis=-1)

    def num_eigenfunctions(self) -> int:
        return self.eigenindicies.shape[0]


class ProductEigenfunctionWithAdditionTheorem(
    EigenfunctionWithAdditionTheorem, ProductEigenfunctions
):
    def __init__(
        self,
        dimensions: List[int],
        eigenindicies: B.Numeric,
        *eigenfunctions: Eigenfunctions,
    ):
        """
        Wrapper class for handling eigenfunctions on product spaces where an addition
        theorem is available.

        Parameters
        ----------
        dimensions : List[int]
            The dimensions of the spaces being producted together
        eigenindicies : B.Numeric
            An array mapping i'th eigenfunction of the product space to
            the index of the eigenfunctions of the subspaces, OR the i'th
            addition level if the eigenfun
        eigenfunctions : Eigenfunctions

        """
        if dimension_indices == None:
            self.dimension_indices = []
            i = 0
            inds = B.linspace(0, sum(dimensions) - 1, sum(dimensions)).astype(int)
            for dim in dimensions:
                self.dimension_indices.append(inds[i : i + dim])
                i += dim
        self.eigenindicies = eigenindicies
        self.level_mapping = [
            num_per_level_to_mapping(eigenfunction.num_eigenfunctions_per_level)
            if isinstance(eigenfunction, EigenfunctionWithAdditionTheorem)
            else [[i] for i in range(eigenindicies.shape[0])]
            for eigenfunction in eigenfunctions
        ]

        self.levelindices = indices_to_levels(
            self.eigenindicies,
            self.level_mapping,
        )
        self.additionfunctions = [
            lambda X, X2, **parameters: eigenfunction._addition_theorem(
                X, X2 ** parameters
            )
            if isinstance(eigenfunction, EigenfunctionWithAdditionTheorem)
            else lambda X, X2, **parameters: eigenfunction(X, **parameters)[
                :, None, ...
            ]
            * eigenfunction(X2, **parameters)[None, ...]
            for eigenfunction in eigenfunctions
        ]
        self.additiondiagfunctions = [
            lambda X, **parameters: eigenfunction._addition_theorem_diag(
                X ** parameters
            )
            if isinstance(eigenfunction, EigenfunctionWithAdditionTheorem)
            else lambda X, **parameters: eigenfunction(X, **parameters) ** 2
            for eigenfunction in eigenfunctions
        ]

        # eigenfunctions are of complete levels only if the level indices map back to the eigenfunctions
        self.complete_levels = np.sort(
            levels_to_indices(
                self.levelindices,
                self.level_mapping,
            )
            == np.sort(self.eigenindicies.shape[0])
        ).all()

        # for consistency between the levels and the eigenindicies for weight filtering.
        if self.complete_levels:
            self.eigenindicies = levels_to_indices(
                self.levelindices,
                self.level_mapping,
            )

        self.eigenfunctions = eigenfunctions

        assert self.eigenindicies.shape[-1] == len(self.eigenfunctions)

    def _addition_theorem(self, X, X2):
        Xs = [B.take(X, inds, axis=-1) for inds in self.dimension_indices]
        X2s = [B.take(X2, inds, axis=-1) for inds in self.dimension_indices]

        addition_funcs = B.stack(
            *[
                additonfunction(X, X2, **parameters)
                for additionfunction, X, X2 in zip(self.additionfunctions, Xs, X2s)
            ],
            axis=-1,
        )

        # TODO: get the right indices, square not vector.
        return addition_funcs[
            :,
            self.levelindices,
            B.linspace(
                0, self.levelindices.shape[1] - 1, self.levelindices.shape[1]
            ).astype(int),
        ].prod(axis=-1)

    def _addition_theorem_diag(self, X, **parameters):
        Xs = [B.take(X, inds, axis=-1) for inds in self.dimension_indices]
        X2s = [B.take(X2, inds, axis=-1) for inds in self.dimension_indices]

        addition_funcs = B.stack(
            *[
                additionfunction(X, **parameters)
                for additionfunction, X in zip(self.additiondiagfunctions, Xs)
            ],
            axis=-1,
        )

        return addition_funcs[
            :,
            self.levelindices,
            B.linspace(
                0, self.levelindices.shape[1] - 1, self.levelindices.shape[1]
            ).astype(int),
        ].prod(axis=-1)

    def num_levels(self):
        assert (
            self.complete_levels
        ), "eigenindicies specified do not correspond to a complete set of levels"
        return self.levelindices.shape[0]

    def num_eigenfunctions_per_level(self):
        assert (
            self.complete_levels
        ), "eigenindicies specified do not correspond to a complete set of levels"
        num_per_level = []

        for i in range(self.levelindices.shape[0]):
            levelindex = self.levelindices[i]

            num_per_level.append(
                int(
                    np.prod(
                        [
                            len(self.level_mapping[d][levelindex[j]])
                            for d in range(self.levelindices.shape[1])
                        ]
                    )
                )
            )


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
            number of eigenfunctions to use for this product space, by default 100
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
        )

        self.sub_space_eigenindicies = find_lowest_sum_combinations(
            sub_space_eigenvalues, self.num_eigen
        )

        self._eigenvalues = sub_space_eigenvalues[
            B.linspace(0, len(self.sub_spaces) - 1, len(self.sub_spaces)).astype(int),
            self.sub_space_eigenindicies[: self.num_eigen, :],
        ].sum(axis=1)

    @property
    def dimension(self) -> int:
        return sum([space.dimension for space in self.sub_spaces])

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        assert num <= self.num_eigen

        max_eigenvalue = self.sub_space_eigenindicies[:num, :].max() + 1

        sub_space_eigenfunctions = [
            space.get_eigenfunctions(max_eigenvalue) for space in self.sub_spaces
        ]

        return ProductEigenfunctions(
            [space.dimension for space in self.sub_spaces],
            self.sub_space_eigenindicies,
            *sub_space_eigenfunctions,
        )

    def get_eigenvalues(self, num: int) -> B.Numeric:
        assert num <= self.num_eigen

        return self._eigenvalues[:num, None]
