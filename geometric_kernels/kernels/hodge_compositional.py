"""
This module provides the :class:`MaternKarhunenLoeveKernel` kernel, the basic
kernel for discrete spectrum spaces, subclasses of :class:`DiscreteSpectrumSpace`.
"""

import lab as B
import numpy as np
from beartype.typing import Dict, Optional

from geometric_kernels.kernels.base import BaseGeometricKernel
from geometric_kernels.kernels.karhunen_loeve import MaternKarhunenLoeveKernel
from geometric_kernels.spaces import HodgeDiscreteSpectrumSpace


class MaternHodgeCompositionalKernel(BaseGeometricKernel):
    r"""
    TODO
    :param space:
        The space to define the kernel upon.
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
            num_levels_per_type = eigenvalues.shape[0]
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
        self.normalize = normalize

    @property
    def space(self) -> HodgeDiscreteSpectrumSpace:
        """
        The space on which the kernel is defined.
        """
        self._space: HodgeDiscreteSpectrumSpace
        return self._space

    def init_params(self) -> Dict[str, B.NPNumeric]:
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

    def eigenvalues(
        self, params: Dict[str, B.Numeric], normalize: Optional[bool] = None
    ) -> B.Numeric:
        """
        Eigenvalues of the kernel.

        :param params:
            Parameters of the kernel. Must contain keys `"lengthscale"` and
            `"nu"`. The shapes of `params["lengthscale"]` and `params["nu"]`
            are `(1,)`.
        :param normalize:
            Whether to normalize kernel to have unit average variance.
            If None, uses `self.normalize` to decide.

            Defaults to None.

        :return:
            An [L, 1]-shaped array.
        """
        assert "harmonic" in params
        assert "gradient" in params
        assert "curl" in params
        assert params["harmonic"]["lengthscale"].shape == (1,)
        assert params["gradient"]["lengthscale"].shape == (1,)
        assert params["curl"]["lengthscale"].shape == (1,)
        assert params["harmonic"]["nu"].shape == (1,)
        assert params["gradient"]["nu"].shape == (1,)
        assert params["curl"]["nu"].shape == (1,)

        spectral_values_harmonic = self.kernel_harmonic.eigenvalues(
            params=params["harmonic"], normalize=normalize
        )
        spectral_values_gradient = self.kernel_gradient.eigenvalues(
            params=params["gradient"], normalize=normalize
        )
        spectral_values_curl = self.kernel_curl.eigenvalues(
            params=params["curl"], normalize=normalize
        )
        spectral_values = {
            "harmonic": spectral_values_harmonic,
            "gradient": spectral_values_gradient,
            "curl": spectral_values_curl,
        }
        return spectral_values

    def K(
        self, params, X: B.Numeric, X2: Optional[B.Numeric] = None, **kwargs  # type: ignore
    ) -> B.Numeric:

        return (
            B.cast(
                B.dtype(self.kernel_harmonic.K(params["harmonic"], X, X2, **kwargs)),
                params["harmonic"]["logit"],
            )
            * self.kernel_harmonic.K(params["harmonic"], X, X2, **kwargs)
            + B.cast(
                B.dtype(self.kernel_harmonic.K(params["gradient"], X, X2, **kwargs)),
                params["gradient"]["logit"],
            )
            * self.kernel_gradient.K(params["gradient"], X, X2, **kwargs)
            + B.cast(
                B.dtype(self.kernel_harmonic.K(params["curl"], X, X2, **kwargs)),
                params["curl"]["logit"],
            )
            * self.kernel_curl.K(params["curl"], X, X2, **kwargs)
        )

    def K_diag(self, params, X: B.Numeric, **kwargs) -> B.Numeric:
        return (
            B.cast(
                B.dtype(self.kernel_harmonic.K_diag(params["harmonic"], X, **kwargs)),
                params["harmonic"]["logit"],
            )
            * self.kernel_harmonic.K_diag(params["harmonic"], X, **kwargs)
            + B.cast(
                B.dtype(self.kernel_harmonic.K_diag(params["gradient"], X, **kwargs)),
                params["gradient"]["logit"],
            )
            * self.kernel_gradient.K_diag(params["gradient"], X, **kwargs)
            + B.cast(
                B.dtype(self.kernel_harmonic.K_diag(params["curl"], X, **kwargs)),
                params["curl"]["logit"],
            )
            * self.kernel_curl.K_diag(params["curl"], X, **kwargs)
        )
