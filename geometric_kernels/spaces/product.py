"""
Implement product spaces
"""

import lab as B
import numpy as np

from .base import DiscreteSpectrumSpace

from geometric_kernels.eigenfunctions import Eigenfunctions


class ProductEigenfunctions(Eigenfunctions):
    def __init__(self, dimensions, eigenindicies, *eigenfunctions):
        self.dimension_indices = []
        i = 0
        inds = B.linspace(0, sum(dimensions) - 1, sum(dimensions)).astype(int)
        for dim in dimensions:
            self.dimension_indices.append(inds[i : i + dim])
            i += dim
        self.eigenindicies = eigenindicies
        self.eigenfunctions = eigenfunctions

    def __call__(self, X, **parameters):
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

    def num_eigenfunctions(self):
        return self.eigenindicies.shape[0]


class ProductDiscreteSpectrumSpace(DiscreteSpectrumSpace):
    def __init__(self, *spaces, num_eigen=100):
        """_summary_

        Parameters
        ----------
        spaces : _type_
            _description_
        num_eigenfunctions : _type_
            _description_
        """
        for space in spaces:
            assert isinstance(space, DiscreteSpectrumSpace)

        self.spaces = spaces
        self.num_eigen = num_eigen

        self.sub_space_eigenindicies = B.stack(
            *[B.zeros(int, self.num_eigen) for space in self.spaces], axis=1
        )

        sub_space_eigenvalues = B.stack(
            *[space.get_eigenvalues(self.num_eigen)[:, 0] for space in self.spaces],
            axis=0,
        )

        # first eigenvalue is the sum of the first eigenvalues of the individual spaces
        curr_sub_space_idx = B.zeros(int, len(self.spaces))[None, :]
        # prebuild array for indexing the first index of the eigenvalue array
        first_index = B.linspace(0, len(self.spaces) - 1, len(self.spaces)).astype(int)
        # perform an iterative search for the smallest eigenvalues.
        # assuming that the eigenvalues come sorted, the next smallest is going to increment
        # on of the subspace eigenindicies by 1.
        i = 0
        while i < self.num_eigen:
            eigenvalues = sub_space_eigenvalues[first_index, curr_sub_space_idx].sum(
                axis=1
            )
            highest_eigenvalue_index = int(eigenvalues.argmin())
            tied_eigenvalues = eigenvalues == eigenvalues[highest_eigenvalue_index]
            tied_eigenvalues_indexes = B.linspace(
                0, len(tied_eigenvalues) - 1, len(tied_eigenvalues)
            ).astype(int)[tied_eigenvalues]
            for index in tied_eigenvalues_indexes:
                self.sub_space_eigenindicies[i, :] = curr_sub_space_idx[index]
                i += 1
                if i >= self.num_eigen:
                    break

            old_indices = curr_sub_space_idx[~tied_eigenvalues]
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
            B.linspace(0, len(self.spaces) - 1, len(self.spaces)).astype(int),
            self.sub_space_eigenindicies[: self.num_eigen, :],
        ].sum(axis=1)

    @property
    def dimension(self):
        return sum([space.dimension for space in self.spaces])

    def get_eigenfunctions(self, num):
        assert num <= self.num_eigen

        max_eigenvalue = self.sub_space_eigenindicies[:num, :].max()

        sub_space_eigenfunctions = [
            space.get_eigenfunctions(max_eigenvalue) for space in self.spaces
        ]

        return ProductEigenfunctions(
            [space.dimension for space in self.spaces],
            self.sub_space_eigenindicies,
            *sub_space_eigenfunctions,
        )

    def get_eigenvalues(self, num):
        assert num <= self.num_eigen

        return self._eigenvalues[:num, None]
