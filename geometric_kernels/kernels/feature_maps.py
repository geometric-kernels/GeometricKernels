"""
Feature maps
"""
from typing import Dict

import lab as B

from geometric_kernels.kernels.geometric_kernels import MaternKarhunenLoeveKernel
from geometric_kernels.lab_extras import float_like, from_numpy
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


def deterministic_feature_map_compact(
    space: DiscreteSpectrumSpace,
    kernel: MaternKarhunenLoeveKernel,
):
    r"""
    Deterministic feature map for compact spaces based on the Laplacian eigendecomposition.

    :return: Callable
        Signature: ``(X, params,  **kwargs)``

        :param X: [N, D] points in the space to evaluate the map on.

        :param params: parameters of the kernel (lengthscale and smoothness).

        :param ``**kwargs``: unused.

        :return: `Tuple(feature_map, context)` where `feature_map` is Callable,
                 and `context` is empty (no context).
    """

    def _map(X: B.Numeric, params, normalize=None, **kwargs) -> B.Numeric:
        """
        Feature map of the Matern kernel defined on DiscreteSpectrumSpace.

        :param X: points in the space to evaluate the map on.
        :param params: parameters of the kernel (lengthscale and smoothness).
        :param normalize: normalize to have unit average variance.
        :param ``**kwargs``: unused.

        :return: `Tuple(features, context)` where `features` is [N, O] features,
                 and `context` is empty (no context).
        """
        repeated_eigenvalues = space.get_repeated_eigenvalues(kernel.num_eigenfunctions)
        spectrum = kernel._spectrum(
            repeated_eigenvalues**0.5,
            nu=params["nu"],
            lengthscale=params["lengthscale"],
        )
        normalize = normalize or (normalize is None and kernel.normalize)
        if normalize:
            normalizer = B.sum(spectrum)
            spectrum = spectrum / normalizer

        weights = B.transpose(B.power(spectrum, 0.5))  # [1, M]
        eigenfunctions = kernel.eigenfunctions(X, **params)  # [N, M]

        _context: Dict[str, str] = {}  # no context
        features = B.cast(float_like(X), eigenfunctions) * B.cast(
            float_like(X), weights
        )  # [N, M]
        return features, _context

    return _map


def random_phase_feature_map_compact(
    space: DiscreteSpectrumSpace,
    kernel: MaternKarhunenLoeveKernel,
    num_random_phases=3000,
):
    r"""
    Random phase feature map for compact spaces based on the Laplacian eigendecomposition.

    :param space: Space.
    :param kernel: kernel.

    :return: Callable
        Signature: ``(X, params, key, **kwargs)``

        :param X: [N, D] points in the space to evaluate the map on.

        :param params: parameters of the kernel (lengthscale and smoothness).

        :param key: random state, either `np.random.RandomState`, `tf.random.Generator`,
                    `torch.Generator` or `jax.tensor` (representing random state).

                     Note that for any backend other than `jax`, passing the same `key`
                     twice does not garantee that the feature map will be the same each time.
                     This is because these backends' random state has... a state.
                     One either has to recreate/restore the state each time or
                     make use of `geometric_kernels.utils.make_deterministic`.

        :param normalize: normalize to have unit average variance.

        :param ``**kwargs``: unused.

        :return: `Tuple(features, context)` where `features` is [N, O] features,
                 and `context` is `{'key': <new key>}`. `<new key>` is the new key
                 for jax, and the same random state (generator) for all other backends.
    """

    def _map(X: B.Numeric, params, key, normalize=None, **kwargs) -> B.Numeric:
        """
        :param X: [N, D] points in the space to evaluate the map on.

        :param params: parameters of the kernel (lengthscale and smoothness).

        :param key: random state, either `np.random.RandomState`, `tf.random.Generator`,
                    `torch.Generator` or `jax.tensor` (representing random state).

                     Note that for any backend other than `jax`, passing the same `key`
                     twice does not garantee that the feature map will be the same each time.
                     This is because these backends' random state has... a state.
                     One either has to recreate/restore the state each time or
                     make use of `geometric_kernels.utils.make_deterministic`.

        :param normalize: normalize to have unit average variance.

        :param ``**kwargs``: unused.

        :return: `Tuple(features, context)` where `features` is [N, O] features,
                 and `context` is `{'key': <new key>}`. `<new key>` is the new key
                 for jax, and the same random state (generator) for all other backends.
        """
        key, random_phases = space.random(key, num_random_phases)  # [O, D]
        eigenvalues = kernel.eigenvalues_laplacian

        spectrum = kernel._spectrum(
            eigenvalues**0.5,
            nu=params["nu"],
            lengthscale=params["lengthscale"],
        )

        weights = B.power(spectrum, 0.5)  # [L, 1]

        random_phases_b = B.cast(float_like(X), from_numpy(X, random_phases))

        phi_product = kernel.eigenfunctions.phi_product(
            X, random_phases_b, **params
        )  # [N, O, L]

        embedding = B.cast(float_like(X), phi_product)  # [N, O, L]
        weights_t = B.cast(float_like(X), B.transpose(weights))

        features = B.reshape(embedding * weights_t, B.shape(X)[0], -1)  # [N, O*L]
        normalize = normalize or (normalize is None and kernel.normalize)
        if normalize:
            normalizer = B.sqrt(B.sum(features**2, axis=-1, squeeze=False))
            features = features / normalizer

        _context: Dict[str, str] = {"key": key}
        return features, _context

    return _map


def random_phase_feature_map_noncompact(
    space: NoncompactSymmetricSpace, num_random_phases=3000
):
    r"""
    Random phase feature map for noncompact symmetric space based on naive algorithm.

    :param space: Space.
    :param num_random_phases: number of random phases to use.
    :param normalize: whether to normalize to have unit variance

    :return: Callable
            Signature: ``(X, params, key, **kwargs)``

            :param X: [N, D] points in the space to evaluate the map on.

            :param params: parameters of the feature map (lengthscale and smoothness).

            :param key: random state, either `np.random.RandomState`, `tf.random.Generator`,
                        `torch.Generator` or `jax.tensor` (representing random state).

                         Note that for any backend other than `jax`, passing the same `key`
                         twice does not garantee that the feature map will be the same each time.
                         This is because these backends' random state has... a state.
                         One either has to recreate/restore the state each time or
                         make use of `geometric_kernels.utils.make_deterministic`.

            :param normalize: normalize to have unit average variance.

            :param ``**kwargs``: unused.

            :return: `Tuple(features, context)` where `features` is [N, O] features,
                     and `context` is `{'key': <new key>}`. `<new key>` is the new key
                     for jax, and the same random state (generator) for all other backends.
    """

    def _map(X: B.Numeric, params, key, normalize=True, **kwargs) -> B.Numeric:
        """
        :param X: [N, D] points in the space to evaluate the map on.

        :param params: parameters of the feature map (lengthscale and smoothness).

        :param key: random state, either `np.random.RandomState`, `tf.random.Generator`,
                    `torch.Generator` or `jax.tensor` (representing random state).

                     Note that for any backend other than `jax`, passing the same `key`
                     twice does not garantee that the feature map will be the same each time.
                     This is because these backends' random state has... a state.
                     One either has to recreate/restore the state each time or
                     make use of `geometric_kernels.utils.make_deterministic`.

        :param normalize: normalize to have unit average variance.

        :param kwargs: unused.

        :return: `Tuple(features, context)` where `features` is [N, O] features,
                 and `context` is `{'key': <new key>}`. `<new key>` is the new key
                 for jax, and the same random state (generator) for all other backends.
        """
        key, random_phases = space.random_phases(key, num_random_phases)  # [O, <axes>]

        key, random_lambda = base_density_sample(
            key,
            (num_random_phases, B.shape(space.rho)[0]),  # [O, D]
            params,
            space.dimension,
            space.rho,
        )  # [O, P]

        # X [N, <axes>]
        random_phases_b = B.expand_dims(
            B.cast(float_like(X), from_numpy(X, random_phases))
        )  # [1, O, <axes>]
        random_lambda_b = B.expand_dims(
            B.cast(float_like(X), from_numpy(X, random_lambda))
        )  # [1, O, P]
        X_b = B.expand_dims(X, axis=-1 - space.num_axes)  # [N, 1, <axes>]

        p = space.power_function(random_lambda_b, X_b, random_phases_b)  # [N, O]
        c = space.inv_harish_chandra(random_lambda_b)  # [1, O]

        out = B.concat(B.real(p) * c, B.imag(p) * c, axis=-1)  # [N, 2*O]
        if normalize:
            normalizer = B.sqrt(B.sum(out**2, axis=-1, squeeze=False))
            out = out / normalizer

        _context: Dict[str, B.types.RandomState] = {"key": key}
        return out, _context

    return _map


def rejection_sampling_feature_map_hyperbolic(
    space: Hyperbolic,
    num_random_phases=3000,
):
    r"""
    Random phase feature map for the Hyperbolic space based on the
    rejection sampling algorithm.

    :param space: Hyperbolic space.
    :param num_random_phases: number of random phases to use.
    :param normalize: whether to normalize to have unit variance.

    :return: Callable
            Signature: ``(X, params, key, **kwargs)``

            :param X: [N, D] points in the space to evaluate the map on.

            :param params: parameters of the feature map (lengthscale and smoothness).

            :param key: random state, either `np.random.RandomState`, `tf.random.Generator`,
                        `torch.Generator` or `jax.tensor` (representing random state).

                         Note that for any backend other than `jax`, passing the same `key`
                         twice does not garantee that the feature map will be the same each time.
                         This is because these backends' random state has... a state.
                         One either has to recreate/restore the state each time or
                         make use of `geometric_kernels.utils.make_deterministic`.

            :param kwargs: unused.

            :return: `Tuple(features, context)` where `features` is [N, O] features,
                     and `context` is `{'key': <new key>}`. `<new key>` is the new key
                     for jax, and the same random state (generator) for all other backends.
    """

    def _map(X: B.Numeric, params, key, normalize=True, **kwargs) -> B.Numeric:
        key, random_phases = space.random_phases(key, num_random_phases)  # [O, D]

        key, random_lambda = hyperbolic_density_sample(
            key, (num_random_phases, B.rank(space.rho)), params, space.dimension
        )  # [O, 1]

        # X [N, D]
        random_phases_b = B.expand_dims(
            B.cast(float_like(X), from_numpy(X, random_phases))
        )  # [1, O, D]
        random_lambda_b = B.expand_dims(
            B.cast(float_like(X), from_numpy(X, random_lambda))
        )  # [1, O, 1]
        X_b = B.expand_dims(X, axis=-2)  # [N, 1, D]

        p = space.power_function(random_lambda_b, X_b, random_phases_b)  # [N, O]

        out = B.concat(B.real(p), B.imag(p), axis=-1)  # [N, 2*O]
        if normalize:
            normalizer = B.sqrt(B.sum(out**2, axis=-1, squeeze=False))
            out = out / normalizer

        _context: Dict[str, B.types.RandomState] = {"key": key}
        return out, _context

    return _map


def rejection_sampling_feature_map_spd(
    space: SymmetricPositiveDefiniteMatrices,
    num_random_phases: int = 3000,
):
    r"""
    Random phase feature map for the SPD space based on the
    rejection sampling algorithm.

    :param space: SymmetricPositiveDefiniteMatrices space.
    :param num_random_phases: number of random phases to use.

    :return: Callable
            Signature: ``(X, params, state, key, **kwargs)``

            :param X: [N, D, D] points in the space to evaluate the map on.

            :param params: parameters of the feature map (lengthscale and smoothness).

            :param key: random state, either `np.random.RandomState`, `tf.random.Generator`,
                        `torch.Generator` or `jax.tensor` (representing random state).

                         Note that for any backend other than `jax`, passing the same `key`
                         twice does not garantee that the feature map will be the same each time.
                         This is because these backends' random state has... a state.
                         One either has to recreate/restore the state each time or
                         make use of `geometric_kernels.utils.make_deterministic`.

            :param kwargs: unused.

            :return: `Tuple(features, context)` where `features` is [N, O] features,
                     and `context` is `{'key': <new key>}`. `<new key>` is the new key
                     for jax, and the same random state (generator) for all other backends.
    """

    def _map(X: B.Numeric, params, key, normalize: bool = True, **kwargs) -> B.Numeric:
        key, random_phases = space.random_phases(key, num_random_phases)  # [O, D, D]

        key, random_lambda = spd_density_sample(
            key, (num_random_phases,), params, space.degree, space.rho
        )  # [O, D]

        # X [N, D, D]
        random_phases_b = B.expand_dims(
            B.cast(B.dtype(X), from_numpy(X, random_phases))
        )  # [1, O, D, D]
        random_lambda_b = B.expand_dims(
            B.cast(B.dtype(X), from_numpy(X, random_lambda))
        )  # [1, O, D]
        X_b = B.expand_dims(X, axis=-3)  # [N, 1, D, D]

        p = space.power_function(random_lambda_b, X_b, random_phases_b)  # [N, O]

        out = B.concat(B.real(p), B.imag(p), axis=-1)  # [N, 2*O]
        if normalize:
            normalizer = B.sqrt(B.sum(out**2, axis=-1, squeeze=False))
            out = out / normalizer

        _context: Dict[str, B.types.RandomState] = {"key": key}
        return out, _context

    return _map
