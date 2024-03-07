"""
Feature maps
"""
import abc
import warnings

import lab as B

from geometric_kernels.kernels.geometric_kernels import MaternKarhunenLoeveKernel
from geometric_kernels.lab_extras import from_numpy
from geometric_kernels.sampling.probability_densities import (
    base_density_sample,
    hyperbolic_density_sample,
    spd_density_sample,
)
from geometric_kernels.spaces import (
    DiscreteSpectrumSpace,
    Hyperbolic,
    NoncompactSymmetricSpace,
    SymmetricPositiveDefiniteMatrices,
)


class FeatureMap(abc.ABC):
    """
    Abstract base class for all feature maps.
    """

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


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
        Feature map of the Matern kernel defined on DiscreteSpectrumSpace.

        :param X: points in the space to evaluate the map on.
        :param params: parameters of the kernel (lengthscale and smoothness).
        :param normalize: normalize to have unit average variance (if omitted
                          or None, follows the standard behavior of
                          MaternKarhunenLoeveKernel).
        :param **kwargs: unused.

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


class RandomPhaseFeatureMapCompact(FeatureMap):
    def __init__(
        self,
        space: DiscreteSpectrumSpace,
        num_levels: int,
        num_random_phases: int = 3000,
    ):
        """
        Random phase feature map for compact spaces based on the Laplacian eigendecomposition.

        :param space: space.
        :param num_levels: number of levels in the kernel approximation.
        :param num_random_phases: number of random phases in generalized random phase Fourier features.
        """
        self.space = space
        self.num_levels = num_levels
        self.num_random_phases = num_random_phases
        self.kernel = MaternKarhunenLoeveKernel(space, num_levels)

    def __call__(
        self, X: B.Numeric, params, *, key, normalize=None, **kwargs
    ) -> B.Numeric:
        """
        :param X: [N, D] points in the space to evaluate the map on.
        :param params: parameters of the kernel (lengthscale and smoothness).
        :param key: random state, either `np.random.RandomState`, `tf.random.Generator`,
                    `torch.Generator` or `jax.tensor` (representing random state).

                     Note that for any backend other than `jax`, passing the same `key`
                     twice does not guarantee that the feature map will be the same each time.
                     This is because these backends' random state has... a state.
                     One either has to recreate/restore the state each time or
                     make use of `geometric_kernels.utils.make_deterministic`.
        :param normalize: normalize to have unit average variance (if omitted
                          or None, follows the standard behavior of
                          MaternKarhunenLoeveKernel).
        :param **kwargs: unused.

        :return: `Tuple(key, features)` where `features` is [N, O] features,
                 and `key` is the new key for `jax`, and the same random
                 state (generator) for all other backends.
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
    def __init__(self, space: NoncompactSymmetricSpace, num_random_phases: int = 3000):
        """
        Random phase feature map for noncompact symmetric space based on naive algorithm.

        :param space: Space.
        :param num_random_phases: number of random phases to use.
        """
        self.space = space
        self.num_random_phases = num_random_phases

    def __call__(
        self, X: B.Numeric, params, *, key, normalize=True, **kwargs
    ) -> B.Numeric:
        """
        :param X: [N, D] points in the space to evaluate the map on.
        :param params: parameters of the feature map (lengthscale and smoothness).
        :param key: random state, either `np.random.RandomState`, `tf.random.Generator`,
                    `torch.Generator` or `jax.tensor` (representing random state).

                     Note that for any backend other than `jax`, passing the same `key`
                     twice does not guarantee that the feature map will be the same each time.
                     This is because these backends' random state has... a state.
                     One either has to recreate/restore the state each time or
                     make use of `geometric_kernels.utils.make_deterministic`.
        :param normalize: normalize to have unit average variance (`True` by default).
        :param **kwargs: unused.

        :return: `Tuple(key, features)` where `features` is [N, O] features,
                 and `key` is the new key for `jax`, and the same random
                 state (generator) for all other backends.
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


class RejectionSamplingFeatureMapHyperbolic(FeatureMap):
    def __init__(self, space: Hyperbolic, num_random_phases: int = 3000):
        """
        Random phase feature map for the Hyperbolic space based on the
        rejection sampling algorithm.

        :param space: Hyperbolic space.
        :param num_random_phases: number of random phases to use.
        """
        self.space = space
        self.num_random_phases = num_random_phases

    def __call__(
        self, X: B.Numeric, params, *, key, normalize=True, **kwargs
    ) -> B.Numeric:
        """
        :param X: [N, D] points in the space to evaluate the map on.
        :param params: parameters of the feature map (lengthscale and smoothness).
        :param key: random state, either `np.random.RandomState`, `tf.random.Generator`,
                    `torch.Generator` or `jax.tensor` (representing random state).

                     Note that for any backend other than `jax`, passing the same `key`
                     twice does not guarantee that the feature map will be the same each time.
                     This is because these backends' random state has... a state.
                     One either has to recreate/restore the state each time or
                     make use of `geometric_kernels.utils.make_deterministic`.
        :param normalize: normalize to have unit average variance (`True` by default).
        :param **kwargs: unused.

        :return: `Tuple(key, features)` where `features` is [N, O] features,
                 and `key` is the new key for `jax`, and the same random
                 state (generator) for all other backends.
        """
        # Default behavior
        if normalize is None:
            normalize = True

        key, random_phases = self.space.random_phases(
            key, self.num_random_phases
        )  # [O, D]

        key, random_lambda = hyperbolic_density_sample(
            key,
            (self.num_random_phases, B.rank(self.space.rho)),
            params,
            self.space.dimension,
        )  # [O, 1]

        random_phases_b = B.expand_dims(
            B.cast(B.dtype(params["lengthscale"]), from_numpy(X, random_phases))
        )  # [1, O, D]
        random_lambda_b = B.expand_dims(
            B.cast(B.dtype(params["lengthscale"]), from_numpy(X, random_lambda))
        )  # [1, O, 1]
        X_b = B.expand_dims(X, axis=-2)  # [N, 1, D]

        p = self.space.power_function(random_lambda_b, X_b, random_phases_b)  # [N, O]

        features = B.concat(B.real(p), B.imag(p), axis=-1)  # [N, 2*O]
        if normalize:
            normalizer = B.sqrt(B.sum(features**2, axis=-1, squeeze=False))
            features = features / normalizer

        return key, features


class RejectionSamplingFeatureMapSpd(FeatureMap):
    def __init__(
        self,
        space: SymmetricPositiveDefiniteMatrices,
        num_random_phases: int = 3000,
    ):
        """
        Random phase feature map for the SPD (Symmetric Positive Definite) space based on the
        rejection sampling algorithm.

        :param space: SymmetricPositiveDefiniteMatrices space.
        :param num_random_phases: number of random phases to use.
        """
        self.space = space
        self.num_random_phases = num_random_phases

    def __call__(
        self, X: B.Numeric, params, *, key, normalize=True, **kwargs
    ) -> B.Numeric:
        """
        :param X: [N, D, D] points in the space to evaluate the map on.
        :param params: parameters of the feature map (lengthscale and smoothness).
        :param key: random state, either `np.random.RandomState`, `tf.random.Generator`,
                    `torch.Generator` or `jax.tensor` (representing random state).

                     Note that for any backend other than `jax`, passing the same `key`
                     twice does not guarantee that the feature map will be the same each time.
                     This is because these backends' random state has... a state.
                     One either has to recreate/restore the state each time or
                     make use of `geometric_kernels.utils.make_deterministic`.
        :param normalize: normalize to have unit average variance (`True` by default).
        :param **kwargs: unused.

        :return: `Tuple(key, features)` where `features` is [N, O] features,
                 and `key` is the new key for `jax`, and the same random
                 state (generator) for all other backends.
        """
        # Default behavior for normalization
        if normalize is None:
            normalize = True

        key, random_phases = self.space.random_phases(
            key, self.num_random_phases
        )  # [O, D, D]

        key, random_lambda = spd_density_sample(
            key, (self.num_random_phases,), params, self.space.degree, self.space.rho
        )  # [O, D]

        random_phases_b = B.expand_dims(
            B.cast(B.dtype(params["lengthscale"]), from_numpy(X, random_phases))
        )  # [1, O, D, D]
        random_lambda_b = B.expand_dims(
            B.cast(B.dtype(params["lengthscale"]), from_numpy(X, random_lambda))
        )  # [1, O, D]
        X_b = B.expand_dims(X, axis=-3)  # [N, 1, D, D]

        p = self.space.power_function(random_lambda_b, X_b, random_phases_b)  # [N, O]

        features = B.concat(B.real(p), B.imag(p), axis=-1)  # [N, 2*O]
        if normalize:
            normalizer = B.sqrt(B.sum(features**2, axis=-1, squeeze=False))
            features = features / normalizer

        return key, features


# compatibility layer
def deprecated(fn):
    """
    Ad-hoc decorator that suggests using new camel-cased classes instead of
    old snake-cased functions.
    """
    fn_name = fn.__name__
    new_name = "".join(word.title() for word in fn_name.split("_"))

    doc = globals()[new_name].__init__.__doc__

    def wrapper(*args, **kwargs):
        warnings.warn(
            f"Calling {fn.__name__} has been deprecated. "
            f"Please use {new_name} instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return fn(*args, **kwargs)

    wrapper.__doc__ = doc
    wrapper.__name__ = fn_name
    return wrapper


@deprecated
def deterministic_feature_map_compact(*args, **kwargs):
    return DeterministicFeatureMapCompact(*args, **kwargs)


@deprecated
def random_phase_feature_map_compact(*args, **kwargs):
    return RandomPhaseFeatureMapCompact(*args, **kwargs)


@deprecated
def random_phase_feature_map_noncompact(*args, **kwargs):
    return RandomPhaseFeatureMapNoncompact(*args, **kwargs)


@deprecated
def rejection_sampling_feature_map_hyperbolic(*args, **kwargs):
    return RejectionSamplingFeatureMapHyperbolic(*args, **kwargs)


@deprecated
def rejection_sampling_feature_map_spd(*args, **kwargs):
    return RejectionSamplingFeatureMapSpd(*args, **kwargs)
