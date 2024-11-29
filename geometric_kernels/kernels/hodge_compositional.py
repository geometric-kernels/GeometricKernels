"""
This module provides the :class:`MaternKarhunenLoeveKernel` kernel, the basic
kernel for discrete spectrum spaces, subclasses of :class:`DiscreteSpectrumSpace`.
"""

import lab as B
import numpy as np
from beartype.typing import Dict, Optional

from geometric_kernels.kernels.base import BaseGeometricKernel
from geometric_kernels.kernels.karhunen_loeve import MaternKarhunenLoeveKernel
from geometric_kernels.lab_extras import from_numpy, is_complex
from geometric_kernels.spaces import HodgeDiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import Eigenfunctions


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
        for type in ["harmonic", "curl", "gradient"]:
            eigenvalues = self.space.get_eigenvalues(self.num_levels, type)
            eigenfunctions = self.space.get_eigenfunctions(self.num_levels, type)
            cur_num_levels = len(eigenvalues)

            setattr(self, f"kernel_{type}", MaternKarhunenLoeveKernel(
                space,
                cur_num_levels,
                normalize,
                eigenvalues=eigenvalues,
                eigenfunctions=eigenfunctions
            ))
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
        TODO
        """
        params = dict(
            logit_h = np.array([1.0]),
            nu_h=np.arr([np.inf]),
            lengthscale_h=np.array([1.0]),
            logit_c = np.array([1.0]),
            nu_c=np.arr([np.inf]),
            lengthscale_c=np.array([1.0]),
            logit_g = np.array([1.0]),
            nu_g=np.arr([np.inf]),
            lengthscale_g=np.array([1.0]),
            )

        return params


    def K(
        self, params, X: B.Numeric, X2: Optional[B.Numeric] = None, **kwargs  # type: ignore
    ) -> B.Numeric:
        # TODO: project params
        return self.kernel_harmonic.K(params, X, X2, **kwargs) + self.kernel_curl.K(params, X, X2, **kwargs) + self.kernel_gradient.K(params, X, X2, **kwargs)


    def K_diag(self, params, X: B.Numeric, **kwargs) -> B.Numeric:
        # TODO: project params
        return self.kernel_harmonic.K_diag(params, X, **kwargs) + self.kernel_curl.K_diag(params, X, **kwargs) + self.kernel_gradient.K_diag(params, X, **kwargs)
