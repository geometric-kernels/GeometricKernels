"""
This module provides the :class:`~.kernels.MaternHodgeCompositionalKernel`
kernel, for discrete spectrum spaces with Hodge decomposition, subclasses
of :class:`~.spaces.HodgeDiscreteSpectrumSpace`.
"""

import lab as B
import numpy as np
from beartype.typing import Dict, Optional

from geometric_kernels.kernels.base import BaseGeometricKernel
from geometric_kernels.kernels.karhunen_loeve import MaternKarhunenLoeveKernel
from geometric_kernels.spaces import HodgeDiscreteSpectrumSpace


class MaternHodgeCompositionalKernel(BaseGeometricKernel):
    r"""
    This class is similar to :class:`~.kernels.MaternKarhunenLoeveKernel`, but
    provides a more expressive family of kernels on the spaces where Hodge
    decomposition is available.

    The resulting kernel is a sum of three kernels,

    .. math:: k(x, x') = a k_{\text{harmonic}}(x, x') + b k_{\text{gradient}}(x, x') + c k_{\text{curl}}(x, x'),

    where $a, b, c$ are weights $a, b, c \geq 0$ and $a + b + c = 1$, and
    $k_{\text{harmonic}}$, $k_{\text{gradient}}$, $k_{\text{curl}}$ are the
    instances of :class:`~.kernels.MaternKarhunenLoeveKernel` that only use the
    eigenpairs of the Laplacian corresponding to a single part of the Hodge
    decomposition.

    The parameters of this kernel are represented by a dict with three keys:
    `"harmonic"`, `"gradient"`, `"curl"`, each corresponding to a dict with
    keys `"logit"`, `"nu"`, `"lengthscale"`, where `"nu"` and `"lengthscale"`
    are the parameters of the respective :class:`~.kernels.MaternKarhunenLoeveKernel`,
    while the `"logit"` parameters determine the weights $a, b, c$ in the
    formula above: $a, b, c$ are the `"logit"` parameters normalized to
    satisfy $a + b + c = 1$.

    Same as for :class:`~.kernels.MaternKarhunenLoeveKernel`, these kernels can sometimes
    be computed more efficiently using addition theorems.

    .. note::
        A brief introduction into the theory behind
        :class:`~.kernels.MaternHodgeCompositionalKernel` can be found in
        :doc:`this </theory/graphs>` documentation page.

    :param space:
        The space to define the kernel upon, a subclass of :class:`~.spaces.HodgeDiscreteSpectrumSpace`.

    :param num_levels:
        Number of levels to include in the summation.

    :param normalize:
        Whether to normalize kernel to have unit average variance.
    """

    def __init__(
        self,
        space: HodgeDiscreteSpectrumSpace,
        num_levels: int,
        normalize: bool = True,
    ):
        super().__init__(space)
        self.num_levels = num_levels  # in code referred to as `L`.
        for hodge_type in ["harmonic", "curl", "gradient"]:
            eigenvalues = self.space.get_eigenvalues(self.num_levels, hodge_type)
            eigenfunctions = self.space.get_eigenfunctions(self.num_levels, hodge_type)
            num_levels_per_type = len(eigenvalues)
            setattr(
                self,
                f"kernel_{hodge_type}",
                MaternKarhunenLoeveKernel(
                    space,
                    num_levels_per_type,
                    normalize,
                    eigenvalues_laplacian=eigenvalues,
                    eigenfunctions=eigenfunctions,
                ),
            )

        self.kernel_harmonic: MaternKarhunenLoeveKernel  # for mypy to know the type
        self.kernel_gradient: MaternKarhunenLoeveKernel  # for mypy to know the type
        self.kernel_curl: MaternKarhunenLoeveKernel  # for mypy to know the type

        self.normalize = normalize

    @property
    def space(self) -> HodgeDiscreteSpectrumSpace:
        """
        The space on which the kernel is defined.
        """
        self._space: HodgeDiscreteSpectrumSpace
        return self._space

    def init_params(self) -> Dict[str, Dict[str, B.NPNumeric]]:
        """
        Initialize the three sets of parameters for the three kernels.
        """
        params = dict(
            harmonic=dict(
                logit=np.array([1.0]),
                nu=np.array([np.inf]),
                lengthscale=np.array([1.0]),
            ),
            gradient=dict(
                logit=np.array([1.0]),
                nu=np.array([np.inf]),
                lengthscale=np.array([1.0]),
            ),
            curl=dict(
                logit=np.array([1.0]),
                nu=np.array([np.inf]),
                lengthscale=np.array([1.0]),
            ),
        )

        return params

    def K(
        self,
        params: Dict[str, Dict[str, B.NPNumeric]],
        X: B.Numeric,
        X2: Optional[B.Numeric] = None,
        **kwargs,
    ) -> B.Numeric:
        """
        Compute the cross-covariance matrix between two batches of vectors of
        inputs, or batches of matrices of inputs, depending on the space.
        """

        assert all(
            key in params for key in ["harmonic", "gradient", "curl"]
        ), "MaternHodgeCompositionalKernel's parameters must contain keys 'harmonic', 'gradient', 'curl'."
        assert all(
            B.shape(params[key]["logit"]) == (1,)
            for key in ["harmonic", "gradient", "curl"]
        ), "The 'logit' parameters of MaternHodgeCompositionalKernel must have shape (1,)."

        # Copy the parameters to avoid modifying the original dict.
        params = {key: params[key].copy() for key in ["harmonic", "gradient", "curl"]}

        coeffs = B.stack(
            *[params[key].pop("logit") for key in ["harmonic", "gradient", "curl"]],
            axis=0,
        )
        coeffs = coeffs / B.sum(coeffs)

        return (
            coeffs[0] * self.kernel_harmonic.K(params["harmonic"], X, X2, **kwargs)
            + coeffs[1] * self.kernel_gradient.K(params["gradient"], X, X2, **kwargs)
            + coeffs[2] * self.kernel_curl.K(params["curl"], X, X2, **kwargs)
        )

    def K_diag(
        self, params: Dict[str, Dict[str, B.NPNumeric]], X: B.Numeric, **kwargs
    ) -> B.Numeric:
        """
        Returns the diagonal of the covariance matrix `self.K(params, X, X)`,
        typically in a more efficient way than actually computing the full
        covariance matrix with `self.K(params, X, X)` and then extracting its
        diagonal.
        """

        assert all(
            key in params for key in ["harmonic", "gradient", "curl"]
        ), "MaternHodgeCompositionalKernel's parameters must contain keys 'harmonic', 'gradient', 'curl'."
        assert all(
            B.shape(params[key]["logit"]) == (1,)
            for key in ["harmonic", "gradient", "curl"]
        ), "The 'logit' parameters of MaternHodgeCompositionalKernel must have shape (1,)."

        # Copy the parameters to avoid modifying the original dict.
        params = {key: params[key].copy() for key in ["harmonic", "gradient", "curl"]}

        coeffs = B.stack(
            *[params[key].pop("logit") for key in ["harmonic", "gradient", "curl"]],
            axis=0,
        )
        coeffs = coeffs / B.sum(coeffs)

        return (
            coeffs[0] * self.kernel_harmonic.K_diag(params["harmonic"], X, **kwargs)
            + coeffs[1] * self.kernel_gradient.K_diag(params["gradient"], X, **kwargs)
            + coeffs[2] * self.kernel_curl.K_diag(params["curl"], X, **kwargs)
        )
