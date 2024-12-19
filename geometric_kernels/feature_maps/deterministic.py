r"""
This module provides the :class:`DeterministicFeatureMapCompact`, a
Karhunen-Loève expansion-based feature map for those
:class:`~.spaces.DiscreteSpectrumSpace`\ s, for which the eigenpairs
are explicitly known.
"""

import lab as B
from beartype.typing import Dict, Optional, Tuple

from geometric_kernels.feature_maps.base import FeatureMap
from geometric_kernels.spaces import DiscreteSpectrumSpace, HodgeDiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import Eigenfunctions


class DeterministicFeatureMapCompact(FeatureMap):
    r"""
    Deterministic feature map for :class:`~.spaces.DiscreteSpectrumSpace`\ s
    for which the actual eigenpairs are explicitly available.

    :param space:
        A :class:`~.spaces.DiscreteSpectrumSpace` space.
    :param num_levels:
        Number of levels in the kernel approximation.
    :param repeated_eigenvalues_laplacian:
        Allowing to pass the repeated eigenvalues of the Laplacian directly,
        instead of computing them from the space. If provided, `eigenfunctions`
        must also be provided. Used for :class:`~.spaces.HodgeDiscreteSpectrumSpace`.
    :param eigenfunctions:
        Allowing to pass the eigenfunctions directly, instead of computing them
        from the space. If provided, `repeated_eigenvalues_laplacian` must also
        be provided. Used for :class:`~.spaces.HodgeDiscreteSpectrumSpace`.
    """

    def __init__(
        self,
        space: DiscreteSpectrumSpace,
        num_levels: int,
        repeated_eigenvalues_laplacian: Optional[B.Numeric] = None,
        eigenfunctions: Optional[Eigenfunctions] = None,
    ):
        self.space = space
        self.num_levels = num_levels

        if repeated_eigenvalues_laplacian is None:
            assert eigenfunctions is None
            repeated_eigenvalues_laplacian = self.space.get_repeated_eigenvalues(
                self.num_levels
            )
            eigenfunctions = self.space.get_eigenfunctions(self.num_levels)
        else:
            assert eigenfunctions is not None
            assert repeated_eigenvalues_laplacian.shape == (num_levels, 1)
            assert eigenfunctions.num_levels == num_levels

        self._repeated_eigenvalues = repeated_eigenvalues_laplacian
        self._eigenfunctions = eigenfunctions

    def __call__(
        self,
        X: B.Numeric,
        params: Dict[str, B.Numeric],
        normalize: bool = True,
        **kwargs,
    ) -> Tuple[None, B.Numeric]:
        """
        :param X:
            [N, ...] points in the space to evaluate the map on.

        :param params:
            Parameters of the kernel (length scale and smoothness).

        :param normalize:
            Normalize to have unit average variance. If omitted, set to True.

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
        from geometric_kernels.kernels.karhunen_loeve import MaternKarhunenLoeveKernel

        spectrum = MaternKarhunenLoeveKernel.spectrum(
            self._repeated_eigenvalues,
            nu=params["nu"],
            lengthscale=params["lengthscale"],
            dimension=self.space.dimension,
        )

        if normalize:
            normalizer = B.sum(spectrum)
            spectrum = spectrum / normalizer

        weights = B.transpose(B.power(spectrum, 0.5))  # [1, M]
        eigenfunctions = self._eigenfunctions(X, **kwargs)  # [N, M]

        features = B.cast(B.dtype(params["lengthscale"]), eigenfunctions) * B.cast(
            B.dtype(params["lengthscale"]), weights
        )  # [N, M]

        return None, features


class HodgeDeterministicFeatureMapCompact(FeatureMap):
    r"""
    Deterministic feature map for :class:`~.spaces.HodgeDiscreteSpectrumSpace`\ s
    for which the actual eigenpairs are explicitly available.

    Corresponds to :class:`~.kernels.MaternHodgeCompositionalKernel` and takes
    parameters in the same format.
    """

    def __init__(self, space: HodgeDiscreteSpectrumSpace, num_levels: int):
        self.space = space
        self.num_levels = num_levels
        for hodge_type in ["harmonic", "curl", "gradient"]:
            repeated_eigenvalues = self.space.get_repeated_eigenvalues(
                self.num_levels, hodge_type
            )
            eigenfunctions = self.space.get_eigenfunctions(self.num_levels, hodge_type)
            num_levels_per_type = len(
                self.space.get_eigenvalues(self.num_levels, hodge_type)
            )
            setattr(
                self,
                f"feature_map_{hodge_type}",
                DeterministicFeatureMapCompact(
                    space,
                    num_levels_per_type,
                    repeated_eigenvalues_laplacian=repeated_eigenvalues,
                    eigenfunctions=eigenfunctions,
                ),
            )

        self.feature_map_harmonic: (
            DeterministicFeatureMapCompact  # for mypy to know the type
        )
        self.feature_map_gradient: (
            DeterministicFeatureMapCompact  # for mypy to know the type
        )
        self.feature_map_curl: (
            DeterministicFeatureMapCompact  # for mypy to know the type
        )

    def __call__(
        self,
        X: B.Numeric,
        params: Dict[str, Dict[str, B.Numeric]],
        normalize: bool = True,
        **kwargs,
    ) -> Tuple[None, B.Numeric]:
        """
        :param X:
            [N, ...] points in the space to evaluate the map on.

        :param params:
            Parameters of the kernel (length scale and smoothness).

        :param normalize:
            Normalize to have unit average variance. If omitted, set to True.

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

        # Copy the parameters to avoid modifying the original dict.
        params = {key: params[key].copy() for key in ["harmonic", "gradient", "curl"]}
        coeffs = B.stack(
            *[params[key].pop("logit") for key in ["harmonic", "gradient", "curl"]],
            axis=0,
        )
        coeffs = coeffs / B.sum(coeffs)
        coeffs = B.sqrt(coeffs)

        return None, B.concat(
            coeffs[0]
            * self.feature_map_harmonic(X, params["harmonic"], normalize, **kwargs)[1],
            coeffs[1]
            * self.feature_map_gradient(X, params["gradient"], normalize, **kwargs)[1],
            coeffs[2]
            * self.feature_map_curl(X, params["curl"], normalize, **kwargs)[1],
            axis=-1,
        )
