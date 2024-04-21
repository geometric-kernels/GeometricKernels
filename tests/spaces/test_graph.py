import warnings

import jax
import lab as B
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import torch

from geometric_kernels.jax import *  # noqa
from geometric_kernels.kernels import MaternKarhunenLoeveKernel
from geometric_kernels.spaces import Graph
from geometric_kernels.tensorflow import *  # noqa
from geometric_kernels.torch import *  # noqa

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

A = np.array(
    [
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
).astype(np.float64)

L = np.array(
    [
        [1, -1, 0, 0, 0, 0, 0],
        [-1, 4, -1, -1, -1, 0, 0],
        [0, -1, 2, 0, 0, -1, 0],
        [0, -1, 0, 2, -1, 0, 0],
        [0, -1, 0, -1, 2, 0, 0],
        [0, 0, -1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
).astype(np.float64)


def run_tests_with_adj(A, L, tol=1e-7, tol_m=1e-4):
    ##############################################
    # Inits

    n = A.shape[0]
    L = B.cast(B.dtype(A), L)
    graph = Graph(A)

    normed_graph = Graph(A, normalize_laplacian=True)

    ##############################################
    # Laplacian computation

    comparison = graph._laplacian == L

    if sp.issparse(comparison):
        comparison = comparison.toarray()

    if isinstance(comparison, np.matrix):  # bug with lab?
        assert comparison.all(), "Laplacian does not match."
    else:
        assert B.all(comparison), "Laplacian does not match."

    normed_l = normed_graph._laplacian
    if sp.issparse(normed_l):
        normed_l = normed_l.toarray()

    assert (
        B.max(B.abs(B.diag(normed_l) - 1)[:-1]) < tol_m
        and B.abs(B.diag(normed_l)[-1] - 0) < tol_m
    )

    ##############################################
    # Eigendecomposition checks

    evecs = graph.get_eigenvectors(n)
    evals = graph.get_eigenvalues(n)

    evals_np, evecs_np = np.linalg.eigh(L)
    evecs_np *= np.sqrt(graph.num_vertices)

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

    normed_evals = normed_graph.get_eigenvalues(n)
    assert (B.min(normed_evals) >= 0) and (
        B.max(normed_evals) <= 2
    )  # well known inequality

    ##############################################
    # Kernel init checks

    K_cons = MaternKarhunenLoeveKernel(graph, n, normalize=False)
    params = K_cons.init_params()

    idx = B.cast(B.dtype(A), np.arange(n)[:, None])

    nu = B.cast(B.dtype(A), np.array([1.0]))
    lscale = B.cast(B.dtype(A), np.array([1.0]))

    K_normed_cons = MaternKarhunenLoeveKernel(normed_graph, n, normalize=False)
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

    nu = B.cast(B.dtype(A), np.array([2.0]))
    params["nu"], params["lengthscale"] = nu, lscale
    Kg = K_cons.K(params, idx)
    K2 = evecs_np @ np.diag(np.power(evals_np + 4, -2)) @ evecs_np.T
    np.testing.assert_allclose(Kg, K2, atol=tol, rtol=tol)

    ##############################################
    # RBF check

    nu = B.cast(B.dtype(A), np.array([np.inf]))
    params["nu"], params["lengthscale"] = nu, lscale
    Kg = K_cons.K(params, idx)
    Ki = evecs_np @ np.diag(np.exp(-0.5 * evals_np)) @ evecs_np.T
    np.testing.assert_allclose(Kg, Ki, atol=tol, rtol=tol)

    ##############################################
    # Fewer than n eigencomps check

    m = 4
    evecs = graph.get_eigenvectors(m)
    evals = graph.get_eigenvalues(m)
    if isinstance(L, jax.numpy.ndarray):
        evals_np, evecs_np = np.linalg.eigh(B.to_numpy(L))
        evals_np, evecs_np = evals_np[:m], evecs_np[:, :m]
    else:
        evals_np, evecs_np = sp.linalg.eigsh(B.to_numpy(L), m, sigma=1e-8)
    evecs_np *= np.sqrt(graph.num_vertices)

    np.testing.assert_allclose(evals[:, 0], evals_np, atol=tol_m, rtol=tol_m)

    try:
        np.testing.assert_allclose(
            np.abs(evecs)[:, :2], np.abs(evecs_np)[:, :2], atol=tol_m, rtol=tol_m
        )
    except AssertionError:
        np.testing.assert_allclose(
            np.abs(evecs)[:, [1, 0]], np.abs(evecs_np)[:, :2], atol=tol_m, rtol=tol_m
        )

    K_cons = MaternKarhunenLoeveKernel(graph, m, normalize=False)
    params = K_cons.init_params()

    nu = B.cast(B.dtype(A), np.array([np.inf]))
    params["nu"], params["lengthscale"] = nu, lscale
    Kg = K_cons.K(params, idx)
    Ki = evecs_np @ np.diag(np.exp(-0.5 * evals_np)) @ evecs_np.T
    np.testing.assert_allclose(Kg, Ki, atol=tol_m, rtol=tol_m)


def test_graphs_numpy():
    run_tests_with_adj(A, L)


def test_graphs_scipy_sparse():
    run_tests_with_adj(sp.csr_matrix(A), L)


def test_graphs_torch():
    run_tests_with_adj(torch.tensor(A), L)


def test_graphs_tf():
    run_tests_with_adj(tf.Variable(A), L)


def test_graphs_jax():
    run_tests_with_adj(jax.numpy.array(A), L, 1e-4, 1e-4)


def test_graphs_torch_cuda():
    if torch.cuda.is_available():
        adj = torch.tensor(A).cuda()

        n = adj.shape[0]
        graph = Graph(adj)
        # normed_graph = Graph(adj, normalize_laplacian=True)  # fails due to bug in lab

        K_cons = MaternKarhunenLoeveKernel(graph, n, normalize=False)
        params = K_cons.init_params()

        params["nu"] = torch.nn.Parameter(torch.tensor([1.0]).cuda())
        params["lengthscale"] = torch.nn.Parameter(torch.tensor([1.0]).cuda())

        idx = torch.arange(n)[:, None].cuda()
        K_cons.K(params, idx)
    else:
        pass
