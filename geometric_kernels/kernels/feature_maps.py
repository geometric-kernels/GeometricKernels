"""
Feature maps
"""
from typing import Dict

import lab as B
from plum import dispatch

from geometric_kernels.kernels import BaseGeometricKernel, MaternKarhunenLoeveKernel
from geometric_kernels.lab_extras import from_numpy
from geometric_kernels.spaces import DiscreteSpectrumSpace, NoncompactSymmetricSpace


@dispatch
def deterministic_feature_map(
    space: DiscreteSpectrumSpace,
    kernel: MaternKarhunenLoeveKernel,
    params,
    state,
):
    assert "eigenvalues_laplacian" in state
    assert "eigenfunctions" in state

    repeated_eigenvalues = space.get_repeated_eigenvalues(kernel.num_eigenfunctions)
    spectrum = kernel._spectrum(
        repeated_eigenvalues**0.5,
        nu=params["nu"],
        lengthscale=params["lengthscale"],
    )

    weights = B.transpose(B.power(spectrum, 0.5))  # [1, M]
    Phi = state["eigenfunctions"]

    def _map(X: B.Numeric) -> B.Numeric:
        eigenfunctions = Phi.__call__(X, **params)  # [N, M]
        return B.cast(B.dtype(X), eigenfunctions) * B.cast(
            B.dtype(X), weights
        )  # [N, M]

    _context: Dict[str, str] = {}  # no context

    return _map, _context


@dispatch
def random_phase_feature_map(
    space: DiscreteSpectrumSpace,
    kernel: MaternKarhunenLoeveKernel,
    params,
    state,
    key,
    order=100,
):
    key, random_phases = space.random(key, order)  # [O, D]
    eigenvalues = state["eigenvalues_laplacian"]

    spectrum = kernel._spectrum(
        eigenvalues**0.5,
        nu=params["nu"],
        lengthscale=params["lengthscale"],
    )

    weights = B.power(spectrum, 0.5)  # [L, 1]
    Phi = state["eigenfunctions"]

    def _map(X: B.Numeric) -> B.Numeric:
        # X [N, D]
        random_phases_b = B.cast(B.dtype(X), from_numpy(X, random_phases))
        embedding = B.cast(
            B.dtype(X), Phi.phi_product(X, random_phases_b, **params)
        )  # [N, O, L]
        weights_t = B.cast(B.dtype(X), B.transpose(weights))
        return B.reshape(embedding * weights_t, B.shape(X)[0], -1)  # [N, O*L]

    _context: Dict[str, str] = {"key": key}

    return _map, _context


@dispatch
def random_phase_feature_map(
    space: NoncompactSymmetricSpace,
    kernel: BaseGeometricKernel,
    params,
    state,
    key,
    order=100
):

    key, random_phases = space.random_phases(key, order)  # [O, D]

    key, random_lambda = spectral_density_sample(key, (order, 1), params, space.dimension)  # [O, 1]

    def _map(X: B.Numeric) -> B.Numeric:
        # X [N, D]
        random_phases_b = B.expand_dims(
            B.cast(B.dtype(X), from_numpy(X, random_phases))
        )  # [1, O, D]
        random_lambda_b = B.expand_dims(
            B.cast(B.dtype(X), from_numpy(X, random_phases))
        )  # [1, O, 1]

        p = space.power_function(random_lambda_b, X[:, None], random_phases_b)  # [N, O]

        c = space.inv_harish_chandra(random_lambda_b)  # [1, O]

        return p * c  # [N, O]

    _context: Dict[str, B.types.RandomState] = {"key": key}

    return _map, _context
