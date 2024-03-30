"""
This module provides the :class:`DeterministicFeatureMapCompact`, a Karhunen-Loève
expansion-based feature map for those :class:`DiscreteSpectrumSpace`s, for
which the eigenpairs are explicitly known.
"""
import lab as B

from geometric_kernels.feature_maps.base import FeatureMap
from geometric_kernels.kernels.karhunen_loeve import MaternKarhunenLoeveKernel
from geometric_kernels.spaces import DiscreteSpectrumSpace


class DeterministicFeatureMapCompact(FeatureMap):
    def __init__(self, space: DiscreteSpectrumSpace, num_levels: int):
        """
        Deterministic feature map for compact spaces based on the Laplacian eigendecomposition.

        :param space: space.
        :param num_levels: number of levels in the kernel approximation.
        """
        self.space = space
        self.num_levels = num_levels
        self.kernel = MaternKarhunenLoeveKernel(space, num_levels)
        self.repeated_eigenvalues = space.get_repeated_eigenvalues(
            self.kernel.num_levels
        )

    def __call__(self, X: B.Numeric, params, normalize=None, **kwargs) -> B.Numeric:
        """
        Feature map of the Matérn kernel defined on DiscreteSpectrumSpace.

        :param X: points in the space to evaluate the map on.
        :param params: parameters of the kernel (lengthscale and smoothness).
        :param normalize: normalize to have unit average variance (if omitted
                          or None, follows the standard behavior of
                          MaternKarhunenLoeveKernel).
        :param ``**kwargs``: unused.

        :return: `Tuple(None, features)` where `features` is [N, O] features.
        """
        spectrum = self.kernel._spectrum(
            self.repeated_eigenvalues**0.5,
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
