"""
This module provides the :class:`ProductDiscreteSpectrumSpace` space and the
respective :class:`~.eigenfunctions.Eigenfunctions` subclass
:class:`ProductEigenfunctions`.

See :doc:`this page </theory/product_spaces>` for a brief account on
theory behind product spaces and the :doc:`Torus.ipynb </examples/Torus>`
notebook for a tutorial on how to use them.
"""

import itertools
import math

import lab as B
import numpy as np
from beartype.typing import List, Optional, Tuple

from geometric_kernels.lab_extras import from_numpy, int_like, take_along_axis
from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import Eigenfunctions
from geometric_kernels.utils.product import make_product, project_product
from geometric_kernels.utils.utils import chain


def _find_lowest_sum_combinations(arr: B.Numeric, k: int) -> B.Numeric:
    """
    For an [N, D] array `arr`, with `arr[i, :]` assumed sorted, find `k`
    smallest sums of one element per each row, i.e. sums of form
    `sum_i arr[i, j_i]`, return the array of indices of the summands.

    :param arr:
        An [N, D]-shaped array.
    :param k:
        Number of smallest sums to find.

    :return:
        A [k, N]-shaped array of the same backend as `arr`.
    """
    N, D = arr.shape

    index_array = B.stack(*[B.zeros(int_like(arr), N) for i in range(k)], axis=0)
    # prebuild array for indexing the first index of the eigenvalue array
    first_index = B.linspace(0, N - 1, N).astype(int)
    i = 0

    # first eigenvalue is the sum of the first eigenvalues of the individual spaces
    curr_idx = B.zeros(int, N)[None, :]
    while i < k:
        # compute eigenvalues of the proposals
        sum_values = arr[first_index, curr_idx].sum(axis=1)

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

        # keep unaccepted ones around
        old_indices = curr_idx[~tied_sums]
        # mutate just accepted ones by adding one to each eigenindex
        new_indices = curr_idx[tied_sums][..., None, :] + B.eye(int, curr_idx.shape[-1])
        new_indices = new_indices.reshape((-1, new_indices.shape[-1]))
        new_indices = B.minimum(new_indices, D - 1)
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


def _num_per_level_to_mapping(num_per_level: List[int]) -> List[List[int]]:
    """
    Given a list of numbers of eigenfunctions per level, return a list `mapping`
    such that `mapping[i][j]` is the index of the `j`-th eigenfunction of the
    `i`-th level.

    :param num_per_level:
        A `num_eigenfunctions_per_level` list of some space.
    """
    mapping = []
    i = 0
    for num in num_per_level:
        mapping.append([i + j for j in range(num)])
        i += num
    return mapping


def _eigenlevelindices_to_eigenfunctionindices(
    eigenlevelindices: B.Numeric, nums_per_level: List[List[int]]
) -> B.Numeric:
    """
    Given an `eigenlevelindices` which maps the product space's level indices to
    the indices of the factor spaces' levels, return an `eigenfunctionindices`
    array that maps the product space's eigenfunction index to the indices of
    the factor spaces' individual eigenfunctions.

    :param eigenlevelindices:
        A [L, S]-shaped array, where `S` is the number of factor spaces and `L`
        is the number of levels, such that `eigenindicies[i, :]` are the indices
        of the levels of the factor spaces that correspond to the `i`'th level
        of the product space.
    :param nums_per_level:
        Each `nums_per_level[j]` represents the `num_eigenfunctions_per_level`
        list of the `j`-th factor space.

    :return:
        A [J, S]-shaped array, where `J` is the total number of eigenfunctions.
    """
    L = eigenlevelindices.shape[0]
    S = eigenlevelindices.shape[1]
    levels_to_eigenfunction_indices: List[List[List[int]]] = [
        _num_per_level_to_mapping(npl) for npl in nums_per_level
    ]  # S lists of lists of length L

    eigenfunctionindices = []
    for cur_level in range(L):
        cur_factor_eigenfunction_indices: List[List[int]] = [
            levels_to_eigenfunction_indices[s][eigenlevelindices[cur_level, s]]
            for s in range(S)
        ]
        new_indices: List[Tuple[int, ...]] = list(
            itertools.product(*cur_factor_eigenfunction_indices)
        )  # part of the resulting eigenfunctionindices for the cur_level level.
        eigenfunctionindices += new_indices

    out = from_numpy(eigenlevelindices, np.array(eigenfunctionindices))
    return out


class ProductEigenfunctions(Eigenfunctions):
    r"""
    Eigenfunctions of the Laplacian on the product space are outer products
    of the eigenfunctions on factors.

    Levels correspond to tuples of levels of the factors.

    :param element_shapes:
        Shapes of the elements in each factor. Can be obtained as properties
        `space.element_shape` of any given factor `space`.
    :param element_dtypes:
        Abstract lab data types of the elements in each factor. Can be obtained
        as properties `space.element_dtype` of any given factor `space`.
    :param eigenindicies:
        A [L, S]-shaped array, where `S` is the number of factor spaces and `L`
        is the number of levels, such that `eigenindicies[i, :]` are the indices
        of the levels of the factor spaces that correspond to the `i`'th level
        of the product space.

        This parameter implicitly determines the number of levels.
    :param ``*eigenfunctions``:
        The :class:`~.Eigenfunctions` subclasses corresponding to each of
        the factor spaces.
    :param dimension_indices:
        Determines how a vector `x` representing a product space element is to
        be mapped into the arrays `xi` that represent elements of the factor
        spaces. `xi` are assumed to be equal to `x[dimension_indices[i]]`,
        possibly up to a reshape. Such a reshape might be necessary to
        accommodate the spaces whose elements are matrices rather than vectors,
        as determined by `element_shapes`. The transformation of `x` into the
        list of `xi`\ s is performed by :func:`~.project_product`.

        If None, assumes the each input is layed-out flattened and concatenated,
        in the same order as the factor spaces. In this case, the inverse to
        :func:`~.project_product` is :func:`~.make_product`.

        Defaults to None.
    """

    def __init__(
        self,
        element_shapes: List[List[int]],
        element_dtypes: List[B.DType],
        eigenindicies: B.Numeric,
        *eigenfunctions: Eigenfunctions,
        dimension_indices: B.Numeric = None,
    ):
        self._num_levels = eigenindicies.shape[0]
        self.element_shapes = element_shapes
        self.element_dtypes = element_dtypes
        dimensions = [math.prod(element_shape) for element_shape in self.element_shapes]
        if dimension_indices is None:
            self.dimension_indices = []
            i = 0
            for dim in dimensions:
                self.dimension_indices.append([*range(i, i + dim)])
                i += dim
        self.eigenindicies = eigenindicies
        self.eigenfunctions = eigenfunctions

        self.nums_per_level: List[List[int]] = [
            eigenfunctions_factor.num_eigenfunctions_per_level
            for eigenfunctions_factor in self.eigenfunctions
        ]  # [S, LFactor] where `LFactor` is eigenfunctions[0].num_levels

        self._eigenfunctionindices = _eigenlevelindices_to_eigenfunctionindices(
            self.eigenindicies, self.nums_per_level
        )

        assert self.eigenindicies.shape[-1] == len(self.eigenfunctions)

    def __call__(self, X: B.Numeric, **kwargs) -> B.Numeric:
        """
        Evaluate the individual eigenfunctions at a batch of input locations.

        .. warning::
            Will `raise NotImplementedError` if some of the factors do not
            implement their `__call__`.

        :param X:
            Points to evaluate the eigenfunctions at, an array of shape [N, D],
            where `N` is the number of points and `D` is the dimension of the
            vectors that represent points in the product space.

            Each point `x` in the product space is a `D`-dimensional vector such
            that `x[dimension_indices[i]]` is the vector that represents the
            flattened element of the `i`-th factor space.

            .. note::
                If the instance was created with `dimension_indices=None`, then
                you can use the :func:`~.make_product` function to convert a
                list of (batches of) points in factor spaces into a (batch of)
                points in the product space.

        :param ``**kwargs``:
            Any additional parameters.

        :return:
            An [N, J]-shaped array, where `J` is the number of eigenfunctions.
        """
        Xs = project_product(
            X, self.dimension_indices, self.element_shapes, self.element_dtypes
        )  # List of S arrays, each of shape [N, *element_shape_s]

        factor_eigenfunction_values = [
            eigenfunction(X, **kwargs)  # [N, Js], Js different for each factor
            for eigenfunction, X in zip(self.eigenfunctions, Xs)
        ]  # List of length S, contains arrays of different shapes (not stackable)

        eigenfunction_values = []
        for j in range(self.num_eigenfunctions):
            m_idx = self._eigenfunctionindices[j]  # (S,)-shaped array
            eigenfunction_values.append(
                B.prod(
                    B.stack(
                        *(
                            factor_eigenfunction_values[s][
                                :, self._eigenfunctionindices[j, s]
                            ]  # [N,]
                            for s in range(len(m_idx))
                        ),
                        axis=-1,
                    ),  # [N, S]
                    axis=1,
                )  # [N,]
            )
        eigenfunction_values = B.stack(*eigenfunction_values, axis=-1)  # [N, J]

        return eigenfunction_values

    @property
    def num_eigenfunctions(self) -> int:
        return self._eigenfunctionindices.shape[0]

    @property
    def num_levels(self) -> int:
        return self._num_levels

    def phi_product(
        self, X: B.Numeric, X2: Optional[B.Numeric] = None, **kwargs
    ) -> B.Numeric:
        if X2 is None:
            X2 = X
        Xs = project_product(
            X, self.dimension_indices, self.element_shapes, self.element_dtypes
        )  # List of S arrays, each of shape [N, *element_shape_s]
        Xs2 = project_product(
            X2, self.dimension_indices, self.element_shapes, self.element_dtypes
        )  # List of S arrays, each of shape [N2, *element_shape_s]

        factor_phi_products = [
            take_along_axis(
                eigenfunction.phi_product(X1, X2, **kwargs),  # [N, N2, L_s]
                from_numpy(X1, self.eigenindicies[None, None, :, s]),
                -1,
            )  # [N, N2, L]
            for s, (eigenfunction, X1, X2) in enumerate(
                zip(self.eigenfunctions, Xs, Xs2)
            )
        ]
        common_dtype = B.promote_dtypes(*[B.dtype(x) for x in factor_phi_products])

        phis = B.stack(
            *[B.cast(common_dtype, x) for x in factor_phi_products], axis=-1
        )  # [N, N2, L, S]

        prod_phis = B.prod(phis, axis=-1)  # [N, N2, L, S] -> [N, N2, L]

        return prod_phis

    def phi_product_diag(self, X: B.Numeric, **kwargs):
        Xs = project_product(
            X, self.dimension_indices, self.element_shapes, self.element_dtypes
        )

        phis = B.stack(
            *[
                take_along_axis(
                    eigenfunction.phi_product_diag(X1, **kwargs),  # [N, L_s]
                    from_numpy(X1, self.eigenindicies[None, :, s]),
                    -1,
                )  # [N, L]
                for s, (eigenfunction, X1) in enumerate(zip(self.eigenfunctions, Xs))
            ],
            axis=-1,
        )  # [N, L, S]

        prod_phis = B.prod(phis, axis=-1)  # [N, L]

        return prod_phis

    @property
    def num_eigenfunctions_per_level(self) -> List[int]:
        L = self.eigenindicies.shape[0]
        S = self.eigenindicies.shape[1]

        totals = []

        for cur_level in range(L):
            totals.append(
                math.prod(
                    self.nums_per_level[s][self.eigenindicies[cur_level, s]]
                    for s in range(S)
                )
            )

        return totals


class ProductDiscreteSpectrumSpace(DiscreteSpectrumSpace):
    r"""
    The GeometricKernels space representing a product

    .. math:: \mathcal{S} = \mathcal{S}_1 \times \ldots \mathcal{S}_S

    of :class:`~.spaces.DiscreteSpectrumSpace`-s $\mathcal{S}_i$.

    Levels are indexed by tuples of levels of the factors.

    .. admonition:: Precomputing optimal levels

        The eigenvalue corresponding to a level of the product space
        represented by a tuple $(j_1, .., j_S)$ is the sum

        .. math:: \lambda^{(1)}_{j_1} + \ldots + \lambda^{(S)}_{j_S}

        of the eigenvalues $\lambda^{(s)}_{j_s}$ of the factors.

        Computing top `num_levels` smallest eigenvalues is thus a combinatorial
        problem, the solution to which we precompute. To make this
        precomputation possible you need to provide the `num_levels` parameter
        when constructing the :class:`ProductDiscreteSpectrumSpace`, unlike for
        other spaces. What is more, the `num` parameter of the
        :meth:`get_eigenfunctions` cannot be larger than the `num_levels` value.

    .. note::
        See :doc:`this page </theory/product_spaces>` for a brief account on
        theory behind product spaces and the :doc:`Torus.ipynb
        </examples/Torus>` notebook for a tutorial on how to use them.

        An alternative to using :class:`ProductDiscreteSpectrumSpace` is to use
        the :class:`~.kernels.ProductGeometricKernel` kernel. The latter is more
        flexible, allowing more general spaces as factors and automatic
        relevance determination -like behavior.

    :param spaces:
        The factors, subclasses of :class:`~.spaces.DiscreteSpectrumSpace`.
    :param num_levels:
        The number of levels to pre-compute for this product space.
    :param num_levels_per_space:
        Number of levels to fetch for each of the factor spaces, to compute
        the product-space levels. This is a single number rather than a list,
        because we currently only support fetching the same number of levels
        for all the factor spaces.

        If not given, `num_levels` levels will be fetched for each factor.

    .. admonition:: Citation

        If you use this GeometricKernels space in your research, please consider
        citing :cite:t:`borovitskiy2020`.
    """

    def __init__(
        self,
        *spaces: DiscreteSpectrumSpace,
        num_levels: int = 25,
        num_levels_per_space: Optional[int] = None,
    ):
        for space in spaces:
            assert isinstance(
                space, DiscreteSpectrumSpace
            ), "One of the spaces is not an instance of DiscreteSpectrumSpace."

        self.factor_spaces = spaces  # List of length S
        self.num_levels = num_levels

        # Perform a breadth-first search for the smallest eigenvalues.
        # Assuming that the eigenvalues come sorted, the next biggest eigenvalue
        # can be found by taking a one-index step in any direction from the
        # current edge of the search space.

        if num_levels_per_space is None:
            num_levels_per_space = num_levels
        assert num_levels <= num_levels_per_space ** len(
            spaces
        ), "Cannot have more levels than there are possible combinations"

        # prefetch the eigenvalues of the subspaces
        factor_space_eigenvalues = B.stack(
            *[
                space.get_eigenvalues(num_levels_per_space)[:, 0]
                for space in self.factor_spaces
            ],
            axis=0,
        )  # [S, num_levels_per_space]

        self.factor_space_eigenindices = _find_lowest_sum_combinations(
            factor_space_eigenvalues, self.num_levels
        )  # [self.num_levels, S]
        self.factor_space_eigenvalues = factor_space_eigenvalues

        self._eigenvalues = factor_space_eigenvalues[
            range(len(self.factor_spaces)),
            self.factor_space_eigenindices,
        ].sum(axis=1)

    def __str__(self):
        return f"ProductDiscreteSpectrumSpace({', '.join(str(space) for space in self.factor_spaces)})"

    @property
    def dimension(self) -> int:
        """
        Returns the dimension of the product space, equal to the sum of
        dimensions of the factors.
        """
        return sum([space.dimension for space in self.factor_spaces])

    def random(self, key, number):
        """
        Sample random points on the product space by concatenating random points
        on the factor spaces via :func:`~.make_product`.

        :param key:
            Either `np.random.RandomState`, `tf.random.Generator`,
            `torch.Generator` or `jax.tensor` (representing random state).
        :param number:
            Number of samples to draw.

        :return:
            An array of `number` uniformly random samples on the space.
        """
        random_points = []
        for factor in self.factor_spaces:
            key, factor_random_points = factor.random(key, number)
            random_points.append(factor_random_points)

        return key, make_product(random_points)

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        Returns the :class:`~.ProductEigenfunctions` object with `num` levels.

        :param num:
            Number of levels. Cannot be larger than the `num_levels` parameter
            of the constructor.
        """
        assert num <= self.num_levels

        max_level = int(self.factor_space_eigenindices[:num, :].max() + 1)

        factor_space_eigenfunctions = [
            space.get_eigenfunctions(max_level) for space in self.factor_spaces
        ]

        return ProductEigenfunctions(
            [space.element_shape for space in self.factor_spaces],
            [space.element_dtype for space in self.factor_spaces],
            self.factor_space_eigenindices[:num],
            *factor_space_eigenfunctions,
        )

    def get_eigenvalues(self, num: int) -> B.Numeric:
        """
        Eigenvalues of the Laplacian corresponding to the first `num` levels.

        :param num:
            Number of levels. Cannot be larger than the `num_levels` parameter
            of the constructor.
        :return:
            (num, 1)-shaped array containing the eigenvalues.
        """
        assert num <= self.num_levels

        return self._eigenvalues[:num, None]

    def get_repeated_eigenvalues(self, num: int) -> B.Numeric:
        """
        Eigenvalues of the Laplacian corresponding to the first `num` levels,
        repeated according to their multiplicity within levels.

        :param num:
            Number of levels. Cannot be larger than the `num_levels` parameter
            of the constructor.

        :return:
            (J, 1)-shaped array containing the repeated eigenvalues,`J is
            the resulting number of the repeated eigenvalues.
        """
        assert num <= self.num_levels

        eigenfunctions = self.get_eigenfunctions(num)
        eigenvalues = self._eigenvalues[:num]
        multiplicities = eigenfunctions.num_eigenfunctions_per_level

        repeated_eigenvalues = chain(eigenvalues, multiplicities)
        return B.reshape(repeated_eigenvalues, -1, 1)  # [J, 1]

    @property
    def element_shape(self):
        """
        :return:
            Sum of the products of the element shapes of the factor spaces.
        """
        return sum(math.prod(space.element_shape) for space in self.factor_spaces)

    @property
    def element_dtype(self):
        """
        :return:
            B.Numeric.
        """
        return B.Numeric
