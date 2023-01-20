"""
Feature maps
"""
from typing import Dict

import lab as B
from plum import dispatch

from geometric_kernels.kernels import MaternKarhunenLoeveKernel
from geometric_kernels.lab_extras import from_numpy
from geometric_kernels.spaces import DiscreteSpectrumSpace


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

    weights = B.power(spectrum, 0.5)  # [M, 1]
    Phi = state["eigenfunctions"]

    def _map(X: B.Numeric) -> B.Numeric:
        eigenfunctions = Phi.__call__(X, **params)  # [N, M]
        return eigenfunctions * weights.T  # [N, M]

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
        random_phases_b = B.cast(from_numpy(random_phases, X), B.dtype(X))
        embedding = Phi.phi_product(X, random_phases_b)  # [N, O, L]
        return B.reshape(embedding * weights.T, B.shape(X)[0], -1)  # [N, O*L]

    _context: Dict[str, str] = {"key": key}

    return _map, _context
