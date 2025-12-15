r"""
This module provides the :class:`MaternKernelHammingGraph` kernel, a subclass of
:class:`MaternKarhunenLoeveKernel` for :class:`HammingGraph` and
:class:`HypercubeGraph` spaces implementing the closed-form formula for the
heat kernel when $\nu = \infty$.
"""

import lab as B
import numpy as np
from beartype.typing import Dict, Optional, Union

from geometric_kernels.kernels.karhunen_loeve import MaternKarhunenLoeveKernel
from geometric_kernels.spaces.eigenfunctions import Eigenfunctions
from geometric_kernels.spaces.hamming_graph import HammingGraph
from geometric_kernels.spaces.hypercube_graph import HypercubeGraph
from geometric_kernels.utils.kernel_formulas.hamming_graph import (
    hamming_graph_heat_kernel,
)
from geometric_kernels.utils.utils import _check_1_vector, _check_field_in_params


class MaternKernelHammingGraph(MaternKarhunenLoeveKernel):
    r"""
    For $\nu = \infty$, there exists a closed-form formula for the heat kernel
    on hamming graphs :class:`HammingGraph` (including the binary hypercube case
    :class:`HypercubeGraph`). This class extends :class:`MaternKarhunenLoeveKernel`
    to implement this formula in the case of $\nu = \infty$ for efficiency.

    .. note::
        We only use the closed form expression if `num_levels` is `d + 1` which
        corresponds to exact computation. When truncated to fewer levels, we
        must use the parent class implementation to ensure consistency with
        feature map approximations.
    """

    def __init__(
        self,
        space: Union[HammingGraph, HypercubeGraph],
        num_levels: int,
        normalize: bool = True,
        eigenvalues_laplacian: Optional[B.Numeric] = None,
        eigenfunctions: Optional[Eigenfunctions] = None,
    ):
        if not isinstance(space, (HammingGraph, HypercubeGraph)):
            raise ValueError(
                f"`space` must be an instance of HammingGraph or HypercubeGraph, but got {type(space)}"
            )

        super().__init__(
            space,
            num_levels,
            normalize,
            eigenvalues_laplacian,
            eigenfunctions,
        )

    def K(
        self,
        params: Dict[str, B.Numeric],
        X: B.Numeric,
        X2: Optional[B.Numeric] = None,
        **kwargs,
    ) -> B.Numeric:
        _check_field_in_params(params, "lengthscale")
        _check_1_vector(params["lengthscale"], 'params["lengthscale"]')

        _check_field_in_params(params, "nu")
        _check_1_vector(params["nu"], 'params["nu"]')

        if B.all(params["nu"] == np.inf):
            d = X.shape[-1]

            # Only use fast path when we have all levels (exact computation)
            if self.num_levels == d + 1:
                # Get q from space (HammingGraph has n_cat, HypercubeGraph is binary q=2)
                q = getattr(self.space, "n_cat", 2)

                return hamming_graph_heat_kernel(params["lengthscale"], X, X2, q=q)

        return super().K(params, X, X2, **kwargs)

    def K_diag(self, params: Dict[str, B.Numeric], X: B.Numeric, **kwargs) -> B.Numeric:
        _check_field_in_params(params, "lengthscale")
        _check_1_vector(params["lengthscale"], 'params["lengthscale"]')

        _check_field_in_params(params, "nu")
        _check_1_vector(params["nu"], 'params["nu"]')

        if B.all(params["nu"] == np.inf):
            d = X.shape[-1]

            # Only use fast path when we have all levels (exact computation)
            if self.num_levels == d + 1:
                return B.ones(B.dtype(params["nu"]), X.shape[0])

        return super().K_diag(params, X, **kwargs)
