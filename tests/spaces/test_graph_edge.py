import warnings

import jax
import lab as B
import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx

from geometric_kernels.jax import *  # noqa
from geometric_kernels.kernels import (
    MaternKarhunenLoeveKernel, 
    MaternKarhunenLoeveKernel_HodgeCompositionEdge
)
from geometric_kernels.spaces.graph_edge import GraphEdge
from geometric_kernels.torch import *  # noqa

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

G = nx.Graph()
G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(1, 5)
G.add_edge(2, 3)
G.add_edge(3, 4)
G.add_edge(4, 5)

triangles = [(1,2,3)]


def run_tests_with_lap(tol=1e-7, tol_m=1e-4):
    ##############################################
    # Inits
    sc = GraphEdge(G,triangle_list=triangles, sc_lifting=True)
    m = sc.num_edges

    ##############################################
    # Laplacian computation
    L = sc.edge_laplacian

    ##############################################
    # Eigendecomposition checks

    evecs = sc.get_eigenvectors(m)
    evals = sc.get_eigenvalues(m)

    evals_np, evecs_np = np.linalg.eigh(L)

    # check vals
    np.testing.assert_allclose(evals[:, 0], evals_np, atol=tol, rtol=tol)

    # check vecs
    np.testing.assert_allclose(
        np.abs(evecs)[:, 2:], np.abs(evecs_np)[:, 2:], atol=tol, rtol=tol
    )

    try:
        np.testing.assert_allclose(
            np.abs(evecs)[:, :2], np.abs(evecs_np)[:, :2], atol=tol, rtol=tol
        )
    except AssertionError:
        np.testing.assert_allclose(
            np.abs(evecs)[:, [1, 0]], np.abs(evecs_np)[:, :2], atol=tol, rtol=tol
        )


    ##############################################
    # Kernel init checks

    K_cons = MaternKarhunenLoeveKernel(sc, m, normalize=False)
    params = K_cons.init_params()

    idx = B.cast(B.dtype(L), np.arange(m)[:, None])

    nu = B.cast(B.dtype(L), np.array([1.0]))
    lscale = B.cast(B.dtype(L), np.array([1.0]))

    K_normed_cons = MaternKarhunenLoeveKernel(sc, m, normalize=False)
    normed_params = K_normed_cons.init_params()

    ##############################################
    # Matern 1 check

    params["nu"], params["lengthscale"] = nu, lscale
    Kg = K_cons.K(params, idx)
    K1 = evecs_np @ np.diag(np.power(evals_np + 2, -1)) @ evecs_np.T
    np.testing.assert_allclose(Kg, K1, atol=tol, rtol=tol)

    normed_params["nu"], normed_params["lengthscale"] = nu, lscale
    K_normed_cons.K(normed_params, idx)

    ##############################################
    # Matern 2 check

    nu = B.cast(B.dtype(L), np.array([2.0]))
    params["nu"], params["lengthscale"] = nu, lscale
    Kg = K_cons.K(params, idx)
    K2 = evecs_np @ np.diag(np.power(evals_np + 4, -2)) @ evecs_np.T
    np.testing.assert_allclose(Kg, K2, atol=tol, rtol=tol)

    ##############################################
    # RBF check

    nu = B.cast(B.dtype(L), np.array([np.inf]))
    params["nu"], params["lengthscale"] = nu, lscale
    Kg = K_cons.K(params, idx)
    Ki = evecs_np @ np.diag(np.exp(-0.5 * evals_np)) @ evecs_np.T
    np.testing.assert_allclose(Kg, Ki, atol=tol, rtol=tol)

    ##############################################
    # Fewer than meigencomps check

    m = 4
    evecs = sc.get_eigenvectors(m)
    evals = sc.get_eigenvalues(m)
    if isinstance(L, jax.numpy.ndarray):
        evals_np, evecs_np = np.linalg.eigh(B.to_numpy(L))
        evals_np, evecs_np = evals_np[:m], evecs_np[:, :m]
    else:
        evals_np, evecs_np = sp.linalg.eigsh(B.to_numpy(L), m, sigma=1e-8)

    np.testing.assert_allclose(evals[:, 0], evals_np, atol=tol_m, rtol=tol_m)

    try:
        np.testing.assert_allclose(
            np.abs(evecs)[:, :2], np.abs(evecs_np)[:, :2], atol=tol_m, rtol=tol_m
        )
    except AssertionError:
        np.testing.assert_allclose(
            np.abs(evecs)[:, [1, 0]], np.abs(evecs_np)[:, :2], atol=tol_m, rtol=tol_m
        )

    K_cons = MaternKarhunenLoeveKernel(sc, m, normalize=False)
    params = K_cons.init_params()

    nu = B.cast(B.dtype(L), np.array([np.inf]))
    params["nu"], params["lengthscale"] = nu, lscale
    Kg = K_cons.K(params, idx)
    Ki = evecs_np @ np.diag(np.exp(-0.5 * evals_np)) @ evecs_np.T
    np.testing.assert_allclose(Kg, Ki, atol=tol_m, rtol=tol_m)


def test_graphs_numpy():
    run_tests_with_lap()

def test_graphs_torch():
    run_tests_with_lap()


def test_graphs_jax():
    run_tests_with_lap(1e-4, 1e-4)


def test_graphs_torch_cuda():
    if torch.cuda.is_available():
        sc = GraphEdge(G,triangle_list=triangles, sc_lifting=True)
        m = sc.num_edges

        K_cons = MaternKarhunenLoeveKernel(sc, m, normalize=False)
        params = K_cons.init_params()

        params["nu"] = torch.nn.Parameter(torch.tensor([1.0]).cuda())
        params["lengthscale"] = torch.nn.Parameter(torch.tensor([1.0]).cuda())

        idx = torch.arange(m)[:, None].cuda()
        K_cons.K(params, idx)
    else:
        pass
