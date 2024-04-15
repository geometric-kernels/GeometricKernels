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
    """
    For an [N, D] array, assumed to be sorted within columns, find k smallest
    sums of one element per each row, return array of indices of the summands.

    Will possibly cause problems if k<D (but unlikely).
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
        # Filter out already searched combinations.
        # See https://stackoverflow.com/a/40056251.
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
    Given a collection of eigenindices [J, S], compute the total multiplicities
    of the corresponding eigenvalues.

    eigidx: [J, S]
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
    Given `eigenindices` which map product space's eigenfunction index to the
    indices of subspaces' eigenlevels, convert them to a mapping of product
    space's eigenfunction index to the indices of subspaces' individual
    eigenfunctions via `nums_per_level`, which gives number of eigenfunctions
    per level for each subspace.

    :return:
        An [J, S]-shaped array, where `J` is the total number of eigenfunctions,
        and `S` is the number of factors.
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
    """
    Eigenfunctions of the Laplacian on the product space are outer products
    of the factors.

    Levels correspond to tuples of levels of the factors.

    :param dimensions:
        The list of dimensions of the factor spaces.
    :param eigenindicies:
        An array mapping a j'th eigenfunction of the product space to the
        index of the eigenlevels of the subspaces.
    :param ``*eigenfunctions``:
        The eigenfunctions of the factor spaces.
    :param dimension_indices:
        Dimension indices.
    """

    def __init__(
        self,
        dimensions: List[int],
        eigenindicies: B.Numeric,
        *eigenfunctions: Eigenfunctions,
        dimension_indices: B.Numeric = None,
    ):
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
        """
        Evaluate the individual eigenfunctions at a batch of input locations.

        .. warning::
            Will `raise NotImplementedError` if some of the factors does not
            implement their `__call__`.

        :param X:
            Points to evaluate the eigenfunctions at, an array of
            shape [N, <axis>], where N is the number of points and <axis> is
            the shape of the arrays that represent the points in a given space.
        :param ``**kwargs``:
            Any additional parameters.

        :return:
            An [N, J]-shaped array, where `J` is the number of eigenfunctions.
        """
        Xs = [B.take(X, inds, axis=-1) for inds in self.dimension_indices]

        eigenfunctions = B.stack(
            *[
                eigenfunction(X, **parameters)  # [N, J]
                for eigenfunction, X in zip(self.eigenfunctions, Xs)
            ],
            axis=-1,
        )  # [N, J, S]

        # eigenindices shape [J, S]

        return eigenfunctions[
            :,
            self._separate_eigenindices,
            B.range(self.eigenindicies.shape[1]),
        ].prod(
            axis=-1
        )  # [N, J, S] --> [N, J]

    @property
    def num_eigenfunctions(self) -> int:
        return self._separate_eigenindices.shape[0]

    @property
    def num_levels(self) -> int:
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
        )  # [N, N2, L, S]

        prod_phis = phis[
            :,
            :,
            self.eigenindicies,
            B.range(self.eigenindicies.shape[1]),
        ].prod(
            axis=-1
        )  # [N, N2, L, S] -> [N, N2, L]

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
    r"""
    The GeometricKernels space representing a product

    .. math:: \mathcal{S} = \mathcal{S}_1 \times \ldots \mathcal{S}_S

    of :class:`~.spaces.DiscreteSpectrumSpace`-s $\mathcal{S}_i$.

    Eigenfunctions on the product space are outer products of the factors'
    eigenfunctions:

    .. math::
        f_{j_1, .., j_S}(x_1, .., x_S) = f^1_{j_1} (x_1) \cdot \ldots \cdot f^S_{j_S} (x_S)

    An eigenfunction above corresponds to the eigenvalue

    .. math::
        \lambda_{j_1, .., j_S} = \lambda^1_{j_1} + \ldots + \lambda^S_{j_S}.

    The *levels* (see :class:`here <.kernels.MaternKarhunenLoeveKernel>` and
    :class:`here <.eigenfunctions.Eigenfunctions>`) on factors define levels on
    the product space. Thus, we operate on levels. Without a further truncation,
    the number of levels on the product space is the product of the numbers of
    levels on the factors, which is typically too many. Thus, an additional
    truncation is needed, i.e. choosing the largest $\lambda_{j_1, .., j_S}$.
    We precompute the optimal truncation leading to the `num_levels` in total.

    .. note::
        A tutorial on how to use this space is available in the
        :doc:`Torus.ipynb </examples/Torus>` notebook.

    :param spaces:
        The factors, subclasses of :class:`~.spaces.DiscreteSpectrumSpace`.
    :param num_levels:
        The number of levels to pre-compute for this product space.
    """

    def __init__(self, *spaces: DiscreteSpectrumSpace, num_levels: int = 25):
        for space in spaces:
            assert isinstance(
                space, DiscreteSpectrumSpace
            ), "One of the spaces is not an instance of DiscreteSpectrumSpace."

        self.sub_spaces = spaces
        self.num_eigen = num_levels

        # Perform a breadth-first search for the smallest eigenvalues.
        # Assuming that the eigenvalues come sorted, the next biggest eigenvalue
        # can be found by taking a one-index step in any direction from the
        # current edge of the search space.

        # prefetch the eigenvalues of the subspaces
        sub_space_eigenvalues = B.stack(
            *[space.get_eigenvalues(self.num_eigen)[:, 0] for space in self.sub_spaces],
            axis=0,
        )  # [L, S]

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
        """
        Returns the dimension of the product space, equal to the sum of
        dimensions of the factors.
        """
        return sum([space.dimension for space in self.sub_spaces])

    def random(self, key, number):
        random_points = []
        for factor in self.sub_spaces:
            key, factor_random_points = factor.random(key, number)
            random_points.append(factor_random_points)

        return key, B.concat(*random_points, axis=1)

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        Returns the :class:`~.ProductEigenfunctions` object with `num` levels.

        :param num:
            Number of levels.
        """
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
