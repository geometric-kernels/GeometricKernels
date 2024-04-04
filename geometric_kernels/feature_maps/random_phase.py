r"""
This module provides the random phase-based feature maps.

Specifically, it provides a random phase-based feature map for
:class:`DiscreteSpectrumSpace <geometric_kernels.spaces.DiscreteSpectrumSpace>`\ s
for which the :doc:`addition theorem </theory/addition_theorem>`-like basis
functions are explicitly available while the actual eigenpairs may remain
implicit.

It also provides a basic random phase-based feature map for
:class:`NoncompactSymmetricSpace <geometric_kernels.spaces.NoncompactSymmetricSpace>`\ s.
It should be used unless a more specialized per-space implementation is
available, like the ones in the module :mod:`geometric_kernels.feature_maps.rejection_sampling`.
"""

import lab as B
from beartype.typing import Dict, Optional, Tuple

from geometric_kernels.feature_maps.base import FeatureMap
from geometric_kernels.feature_maps.probability_densities import base_density_sample
from geometric_kernels.kernels.karhunen_loeve import MaternKarhunenLoeveKernel
from geometric_kernels.lab_extras import from_numpy
from geometric_kernels.spaces import DiscreteSpectrumSpace, NoncompactSymmetricSpace


class RandomPhaseFeatureMapCompact(FeatureMap):
    r"""
    Random phase feature map for :class:`DiscreteSpectrumSpace
    <geometric_kernels.spaces.base.DiscreteSpectrumSpace>`\ s for which the
    :doc:`addition theorem </theory/addition_theorem>`-like basis functions are
    explicitly available while the actual eigenpairs may remain implicit.

    :param space: a :class:`DiscreteSpectrumSpace
        <geometric_kernels.spaces.base.DiscreteSpectrumSpace>` space.
    :param num_levels: number of levels in the kernel approximation.
    :param num_random_phases: number of random phases used in the generalized
        random phase Fourier features technique.
    """

    def __init__(
        self,
        space: DiscreteSpectrumSpace,
        num_levels: int,
        num_random_phases: int = 3000,
    ):
        self.space = space
        self.num_levels = num_levels
        self.num_random_phases = num_random_phases
        self.kernel = MaternKarhunenLoeveKernel(space, num_levels)

    def __call__(
        self,
        X: B.Numeric,
        params: Dict[str, B.Numeric],
        *,
        key: B.RandomState,
        normalize: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[B.RandomState, B.Numeric]:
        """
        :param X: [N, ...] points in the space to evaluate the map on.

        :param params: parameters of the kernel (length scale and smoothness).

        :param key: random state, either `np.random.RandomState`,
            `tf.random.Generator`, `torch.Generator` or `jax.tensor` (which
            represents a random state).

            .. note::
                For any backend other than `jax`, passing the same `key` twice
                does not guarantee that the feature map will be the same each
                time. This is because these backends' random state has... a
                state. To evaluate the same (including randomness) feature map
                on different inputs, you can either save/restore state manually
                each time or use the helper function
                :func:`geometric_kernels.utils.utils.make_deterministic` which
                does this for you.

        :param normalize: normalize to have unit average variance (if omitted
            or None, follows the standard behavior of
            :class:`MaternKarhunenLoeveKernel
            <geometric_kernels.kernels.MaternKarhunenLoeveKernel>`).
        :param ``**kwargs``: unused.

        :return: `Tuple(key, features)` where `features` is an [N, O] array, N
            is the number of inputs and O is the dimension of the feature map;
            `key` is the updated random key for `jax`, or the similar random
            state (generator) for any other backends.
        """
        key, random_phases = self.space.random(key, self.num_random_phases)  # [O, D]
        eigenvalues = self.kernel.eigenvalues_laplacian

        spectrum = self.kernel._spectrum(
            eigenvalues**0.5,
            nu=params["nu"],
            lengthscale=params["lengthscale"],
        )

        weights = B.power(spectrum, 0.5)  # [L, 1]

        random_phases_b = B.cast(
            B.dtype(params["lengthscale"]), from_numpy(X, random_phases)
        )

        phi_product = self.kernel.eigenfunctions.phi_product(
            X, random_phases_b, **params
        )  # [N, O, L]

        embedding = B.cast(B.dtype(params["lengthscale"]), phi_product)  # [N, O, L]
        weights_t = B.cast(B.dtype(params["lengthscale"]), B.transpose(weights))

        features = B.reshape(embedding * weights_t, B.shape(X)[0], -1)  # [N, O*L]
        normalize = normalize or (normalize is None and self.kernel.normalize)
        if normalize:
            normalizer = B.sqrt(B.sum(features**2, axis=-1, squeeze=False))
            features = features / normalizer

        return key, features


class RandomPhaseFeatureMapNoncompact(FeatureMap):
    r"""
    Basic random phase feature map for :class`NoncompactSymmetricSpace
    <geometric_kernels.spaces.NoncompactSymmetricSpace>`\ s (based on
    the importance sampling).

    This feature map should not be used if a space-specific alternative exists.

    :param space: a :class:`NoncompactSymmetricSpace
        <geometric_kernels.spaces.base.NoncompactSymmetricSpace>` space.
    :param num_random_phases: number of random phases to use.
    """

    def __init__(self, space: NoncompactSymmetricSpace, num_random_phases: int = 3000):
        self.space = space
        self.num_random_phases = num_random_phases

    def __call__(
        self,
        X: B.Numeric,
        params: Dict[str, B.Numeric],
        *,
        key: B.RandomState,
        normalize: Optional[bool] = True,
        **kwargs,
    ) -> Tuple[B.RandomState, B.Numeric]:
        """
        :param X: [N, ...] points in the space to evaluate the map on.
        :param params: parameters of the feature map (length scale and smoothness).
        :param key: random state, either `np.random.RandomState`,
            `tf.random.Generator`, `torch.Generator` or `jax.tensor` (which
            represents a random state).

            .. note::
                For any backend other than `jax`, passing the same `key` twice
                does not guarantee that the feature map will be the same each
                time. This is because these backends' random state has... a
                state. To evaluate the same (including randomness) feature map
                on different inputs, you can either save/restore state manually
                each time or use the helper function
                :func:`geometric_kernels.utils.utils.make_deterministic` which
                does this for you.

        :param normalize: normalize to have unit average variance (`True` by default).
        :param ``**kwargs``: unused.

        :return: `Tuple(key, features)` where `features` is an [N, O] array, N
            is the number of inputs and O is the dimension of the feature map;
            `key` is the updated random key for `jax`, or the similar random
            state (generator) for any other backends.
        """

        # default behavior
        if normalize is None:
            normalize = True

        key, random_phases = self.space.random_phases(
            key, self.num_random_phases
        )  # [O, <axes>]

        key, random_lambda = base_density_sample(
            key,
            (self.num_random_phases, B.shape(self.space.rho)[0]),  # [O, D]
            params,
            self.space.dimension,
            self.space.rho,
        )  # [O, P]

        random_phases_b = B.expand_dims(
            B.cast(B.dtype(params["lengthscale"]), from_numpy(X, random_phases))
        )  # [1, O, <axes>]
        random_lambda_b = B.expand_dims(
            B.cast(B.dtype(params["lengthscale"]), from_numpy(X, random_lambda))
        )  # [1, O, P]
        X_b = B.expand_dims(X, axis=-1 - self.space.num_axes)  # [N, 1, <axes>]

        p = self.space.power_function(random_lambda_b, X_b, random_phases_b)  # [N, O]
        c = self.space.inv_harish_chandra(random_lambda_b)  # [1, O]

        features = B.concat(B.real(p) * c, B.imag(p) * c, axis=-1)  # [N, 2*O]
        if normalize:
            normalizer = B.sqrt(B.sum(features**2, axis=-1, squeeze=False))
            features = features / normalizer

        return key, features
