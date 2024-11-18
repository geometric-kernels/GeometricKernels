"""
This module provides the :class:`ProductGeometricKernel` kernel for
constructing product kernels from a sequence of kernels.

See :doc:`this page </theory/product_kernels>` for a brief account on
theory behind product kernels and the :doc:`Torus.ipynb </examples/Torus>`
notebook for a tutorial on how to use them.
"""

import math

import lab as B
from beartype.typing import Dict, List, Optional

from geometric_kernels.kernels.base import BaseGeometricKernel
from geometric_kernels.spaces import Space
from geometric_kernels.utils.product import params_to_params_list, project_product


class ProductGeometricKernel(BaseGeometricKernel):
    r"""
    Product kernel, defined as the product of a sequence of kernels.

    See :doc:`this page </theory/product_kernels>` for a brief account on
    theory behind product kernels and the :doc:`Torus.ipynb </examples/Torus>`
    notebook for a tutorial on how to use them.

    :param ``*kernels``:
        A sequence of kernels to compute the product of. Cannot contain another
        instance of :class:`ProductGeometricKernel`. We denote the number of
        factors, i.e. the length of the "sequence", by s.
    :param dimension_indices:
        Determines how a product kernel input vector `x` is to be mapped into
        the inputs `xi` for the factor kernels. `xi` are assumed to be equal to
        `x[dimension_indices[i]]`, possibly up to a reshape. Such a reshape
        might be necessary to accommodate the spaces whose elements are matrices
        rather than vectors, as determined by `element_shapes`. The
        transformation of `x` into the list of `xi`\ s is performed
        by :func:`~.project_product`.

        If None, assumes the each input is layed-out flattened and concatenated,
        in the same order as the factor spaces. In this case, the inverse to
        :func:`~.project_product` is :func:`~.make_product`.

        Defaults to None.

    .. note::
        `params` of a :class:`ProductGeometricKernel` are such that
        `params["lengthscale"]` and `params["nu"]` are (s,)-shaped arrays, where
        `s` is the number of factors.

        Basically, `params["lengthscale"][i]` stores the length scale parameter
        for the `i`-th factor kernel. Same goes for `params["nu"]`. Importantly,
        this enables *automatic relevance determination*-like behavior.
    """

    def __init__(
        self,
        *kernels: BaseGeometricKernel,
        dimension_indices: Optional[List[List[int]]] = None,
    ):
        self.kernels = kernels
        self.spaces: List[Space] = []
        for kernel in self.kernels:
            # Make sure there is no product kernel in the list of kernels.
            assert isinstance(kernel.space, Space)
            self.spaces.append(kernel.space)
        self.element_shapes = [space.element_shape for space in self.spaces]
        self.element_dtypes = [space.element_dtype for space in self.spaces]

        if dimension_indices is None:
            dimensions = [math.prod(shape) for shape in self.element_shapes]
            self.dimension_indices: List[List[int]] = []
            i = 0
            inds = [*range(sum(dimensions))]
            for dim in dimensions:
                self.dimension_indices.append(inds[i : i + dim])
                i += dim
        else:
            assert len(dimension_indices) == len(self.kernels)
            for idx_list in dimension_indices:
                assert all(idx >= 0 for idx in idx_list)

            self.dimension_indices = dimension_indices

    @property
    def space(self) -> List[Space]:
        """
        The list of spaces upon which the factor kernels are defined.
        """
        return self.spaces

    def init_params(self) -> Dict[str, B.NPNumeric]:
        r"""
        Returns a dict `params` where `params["lengthscale"]` is the
        concatenation of all `self.kernels[i].init_params()["lengthscale"]` and
        same for `params["nu"]`.
        """
        nu_list: List[B.NPNumeric] = []
        lengthscale_list: List[B.NPNumeric] = []

        for kernel in self.kernels:
            cur_params = kernel.init_params()
            assert cur_params["lengthscale"].shape == (1,)
            assert cur_params["nu"].shape == (1,)
            nu_list.append(cur_params["nu"])
            lengthscale_list.append(cur_params["lengthscale"])

        params: Dict[str, B.NPNumeric] = {}
        params["nu"] = B.concat(*nu_list)
        params["lengthscale"] = B.concat(*lengthscale_list)
        return params

    def K(self, params: Dict[str, B.Numeric], X, X2=None, **kwargs) -> B.Numeric:
        if X2 is None:
            X2 = X

        Xs = project_product(
            X, self.dimension_indices, self.element_shapes, self.element_dtypes
        )
        X2s = project_product(
            X2, self.dimension_indices, self.element_shapes, self.element_dtypes
        )
        params_list = params_to_params_list(len(self.kernels), params)

        return B.prod(
            B.stack(
                *[
                    kernel.K(p, X, X2)
                    for kernel, X, X2, p in zip(self.kernels, Xs, X2s, params_list)
                ],
                axis=-1,
            ),
            axis=-1,
        )

    def K_diag(self, params, X):
        Xs = project_product(
            X, self.dimension_indices, self.element_shapes, self.element_dtypes
        )
        params_list = params_to_params_list(len(self.kernels), params)

        return B.prod(
            B.stack(
                *[
                    kernel.K_diag(p, X)
                    for kernel, X, p in zip(self.kernels, Xs, params_list)
                ],
                axis=-1,
            ),
            axis=-1,
        )
