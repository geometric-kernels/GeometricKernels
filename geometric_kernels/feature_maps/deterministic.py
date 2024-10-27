r"""
This module provides the :class:`DeterministicFeatureMapCompact`, a
Karhunen-Loève expansion-based feature map for those
:class:`~.spaces.DiscreteSpectrumSpace`\ s, for which the eigenpairs
are explicitly known.
"""

import lab as B
from beartype.typing import Dict, Optional, Tuple

from geometric_kernels.feature_maps.base import FeatureMap
from geometric_kernels.spaces import DiscreteSpectrumSpace


class DeterministicFeatureMapCompact(FeatureMap):
    r"""
    Deterministic feature map for :class:`~.spaces.DiscreteSpectrumSpace`\ s
    for which the actual eigenpairs are explicitly available.

    :param space:
        A :class:`~.spaces.DiscreteSpectrumSpace` space.
    :param num_levels:
        Number of levels in the kernel approximation.
    """

    def __init__(self, space: DiscreteSpectrumSpace, num_levels: int):
        from geometric_kernels.kernels.karhunen_loeve import MaternKarhunenLoeveKernel

        self.space = space
        self.num_levels = num_levels
        self.kernel = MaternKarhunenLoeveKernel(space, num_levels)
        self.repeated_eigenvalues = space.get_repeated_eigenvalues(
            self.kernel.num_levels
        )

    def __call__(
        self,
        X: B.Numeric,
        params: Dict[str, B.Numeric],
        normalize: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[None, B.Numeric]:
        """
        :param X:
            [N, ...] points in the space to evaluate the map on.
        :param params:
            Parameters of the kernel (length scale and smoothness).
        :param normalize:
            Normalize to have unit average variance (if omitted
            or None, follows the standard behavior of
            :class:`~.kernels.MaternKarhunenLoeveKernel`).
        :param ``**kwargs``:
            Unused.

        :return:
            `Tuple(None, features)` where `features` is an [N, O] array, N
            is the number of inputs and O is the dimension of the feature map.

        .. note::
           The first element of the returned tuple is the simple None and
           should be ignored. It is only there to support the abstract
           interface: for some other subclasses of :class:`FeatureMap`, this
           first element may be an updated random key.
        """
        spectrum = self.kernel._spectrum(
            self.repeated_eigenvalues,
            nu=params["nu"],
            lengthscale=params["lengthscale"],
        )
        normalize = normalize or (normalize is None and self.kernel.normalize)
        if normalize:
            normalizer = B.sum(spectrum)
            spectrum = spectrum / normalizer

        weights = B.transpose(B.power(spectrum, 0.5))  # [1, M]
        eigenfunctions = self.kernel.eigenfunctions(X, **params)  # [N, M]

        features = B.cast(B.dtype(params["lengthscale"]), eigenfunctions) * B.cast(
            B.dtype(params["lengthscale"]), weights
        )  # [N, M]
        return None, features
