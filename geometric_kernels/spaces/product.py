"""
Implement product spaces
"""

from typing import List

import lab as B
import numpy as np

from .base import DiscreteSpectrumSpace

from geometric_kernels.eigenfunctions import Eigenfunctions


class ProductEigenfunctions(Eigenfunctions):
    def __init__(
        self,
        dimensions: List[int],
        eigenindicies: B.Numeric,
        *eigenfunctions: Eigenfunctions,
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


class ProductDiscreteSpectrumSpace(DiscreteSpectrumSpace):
    def __init__(self, *spaces: DiscreteSpectrumSpace, num_eigen: int = 100):
        """Implementation of products of discrete spectrum spaces.
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

        self.sub_space_eigenindicies = B.stack(
            *[B.zeros(int, self.num_eigen) for space in self.sub_spaces], axis=1
        )

        # perform an breadth-first search for the smallest eigenvalues,
        # assuming that the eigenvalues come sorted,the next biggest eigenvalue
        # can be found by taking a one-index step in any direction from the current
        # edge of the searchspace

        # prefetch the eigenvalues of the subspaces
        sub_space_eigenvalues = B.stack(
            *[space.get_eigenvalues(self.num_eigen)[:, 0] for space in self.sub_spaces],
            axis=0,
        )
        # prebuild array for indexing the first index of the eigenvalue array
        first_index = B.linspace(
            0, len(self.sub_spaces) - 1, len(self.sub_spaces)
        ).astype(int)
        i = 0

        # first eigenvalue is the sum of the first eigenvalues of the individual spaces
        curr_sub_space_idx = B.zeros(int, len(self.sub_spaces))[None, :]
        while i < self.num_eigen:
            # compute eigenvalues of the proposals
            eigenvalues = sub_space_eigenvalues[first_index, curr_sub_space_idx].sum(
                axis=1
            )

            # Compute tied smallest new eigenvalues
            highest_eigenvalue_index = int(eigenvalues.argmin())
            tied_eigenvalues = eigenvalues == eigenvalues[highest_eigenvalue_index]
            tied_eigenvalues_indexes = B.linspace(
                0, len(tied_eigenvalues) - 1, len(tied_eigenvalues)
            ).astype(int)[tied_eigenvalues]

            # Add new eigenvalues to indexing array
            for index in tied_eigenvalues_indexes:
                self.sub_space_eigenindicies[i, :] = curr_sub_space_idx[index]
                i += 1
                if i >= self.num_eigen:
                    break

            # create new proposal eigenindicies

            # keep unaccepted ones around
            old_indices = curr_sub_space_idx[~tied_eigenvalues]
            # mutate just accepted ones by adding one to each eigenindex
            new_indices = curr_sub_space_idx[tied_eigenvalues][..., None, :] + B.eye(
                int, curr_sub_space_idx.shape[-1]
            )
            new_indices = new_indices.reshape((-1, new_indices.shape[-1]))
            curr_sub_space_idx = B.concat(old_indices, new_indices, axis=0)
            curr_sub_space_idx = np.unique(
                B.to_numpy(
                    curr_sub_space_idx.reshape((-1, curr_sub_space_idx.shape[-1]))
                ),
                axis=0,
            )
            # Filter out already searched combinations. See accepted answer of https://stackoverflow.com/questions/40055835/removing-elements-from-an-array-that-are-in-another-array
            dims = (
                np.maximum(
                    curr_sub_space_idx.max(0), self.sub_space_eigenindicies.max(0)
                )
                + 1
            )
            curr_sub_space_idx = curr_sub_space_idx[
                ~np.in1d(
                    np.ravel_multi_index(curr_sub_space_idx.T, dims),
                    np.ravel_multi_index(self.sub_space_eigenindicies.T, dims),
                )
            ]

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
