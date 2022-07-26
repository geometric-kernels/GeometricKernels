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
        [0, 1, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ]
).astype("float")

L = np.array(
    [
        [1, -1, 0, 0, 0, 0],
        [-1, 4, -1, -1, -1, 0],
        [0, -1, 2, 0, 0, -1],
        [0, -1, 0, 2, -1, 0],
        [0, -1, 0, -1, 2, 0],
        [0, 0, -1, 0, 0, 1],
    ]
).astype("float")


def run_tests_with_adj(A, L, tol=1e-14, tol_m=1e-7):
    ##############################################
    # Inits

    n = A.shape[0]
    L = B.cast(B.dtype(A), L)
    graph = Graph(A)

    ##############################################
    # Laplacian computation

    comparison = graph._laplacian == L

    if sp.issparse(comparison):
        comparison = comparison.toarray()

    if type(comparison) == np.matrix:  # bug with lab?
        assert comparison.all(), "Laplacian does not match."
    else:
        assert B.all(comparison), "Laplacian does not match."

    ##############################################
    # Eigendecomposition checks

    evecs = graph.get_eigenvectors(n)
    evals = graph.get_eigenvalues(n)

    evals_np, evecs_np = np.linalg.eigh(L)

    # check vals
    np.testing.assert_allclose(evals[:, 0], evals_np, atol=tol, rtol=tol)

    # check vecs
    np.testing.assert_allclose(np.abs(evecs), np.abs(evecs_np), atol=tol, rtol=tol)

    ##############################################
    # Kernel init checks

    K_cons = MaternKarhunenLoeveKernel(graph, n)
    params, state = K_cons.init_params_and_state()
    idx = B.cast(B.dtype(A), np.arange(n)[:, None])

    nu = B.cast(B.dtype(A), np.array([1.0]))
    lscale = B.cast(B.dtype(A), np.array([1.0]))

    ##############################################
    # Matern 1 check

    params["nu"], params["lengthscale"] = nu, lscale
    Kg = K_cons.K(params, state, idx)
    K1 = evecs_np @ np.diag(np.power(evals_np + 2, -1)) @ evecs_np.T
    np.testing.assert_allclose(Kg, K1, atol=tol, rtol=tol)

    ##############################################
    # Matern 2 check

    nu = B.cast(B.dtype(A), np.array([2.0]))
    params["nu"], params["lengthscale"] = nu, lscale
    Kg = K_cons.K(params, state, idx)
    K2 = evecs_np @ np.diag(np.power(evals_np + 4, -2)) @ evecs_np.T
    np.testing.assert_allclose(Kg, K2, atol=tol, rtol=tol)

    ##############################################
    # RBF check

    nu = B.cast(B.dtype(A), np.array([np.inf]))
    params["nu"], params["lengthscale"] = nu, lscale
    Kg = K_cons.K(params, state, idx)
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

    np.testing.assert_allclose(evals[:, 0], evals_np, atol=tol_m, rtol=tol_m)
    np.testing.assert_allclose(np.abs(evecs), np.abs(evecs_np), atol=tol_m, rtol=tol_m)

    K_cons = MaternKarhunenLoeveKernel(graph, m)
    params, state = K_cons.init_params_and_state()

    nu = B.cast(B.dtype(A), np.array([np.inf]))
    params["nu"], params["lengthscale"] = nu, lscale
    Kg = K_cons.K(params, state, idx)
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
    run_tests_with_adj(jax.numpy.array(A), L, 1e-6, 1e-6)
