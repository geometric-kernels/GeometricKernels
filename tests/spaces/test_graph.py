import numpy as np
import scipy.sparse as sp

from geometric_kernels.kernels import MaternKarhunenLoeveKernel
from geometric_kernels.lab_extras import degree
from geometric_kernels.spaces import Graph

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


def run_tests_with_adj(A):
    ##############################################
    # Inits

    n = A.shape[0]
    graph = Graph(A)

    ##############################################
    # Laplacian computation

    L = degree(A) - A
    comparison = graph._laplacian == L

    if sp.issparse(comparison):
        comparison = comparison.toarray()
    assert comparison.all(), "Laplacian does not match."

    ##############################################
    # Eigendecomposition checks

    evecs = graph.get_eigenvectors(n)
    evals = graph.get_eigenvalues(n)

    evals_np, evecs_np = np.linalg.eigh(L if not sp.issparse(L) else L.toarray())

    # check vals
    np.testing.assert_allclose(evals[:, 0], evals_np, atol=1e-14, rtol=1e-14)

    # check vecs
    np.testing.assert_allclose(np.abs(evecs), np.abs(evecs_np), atol=1e-14, rtol=1e-14)

    ##############################################
    # Kernel init checks

    K_cons = MaternKarhunenLoeveKernel(graph, n)
    params, state = K_cons.init_params_and_state()
    idx = np.arange(n)[:, None]

    ##############################################
    # Matern 1 check

    params["nu"] = np.array([1.0])
    params["lengthscale"] = np.array([1.0])
    Kg = K_cons.K(params, state, idx)
    K1 = evecs_np @ np.diag(np.power(evals_np + 2, -1)) @ evecs_np.T
    np.testing.assert_allclose(Kg, K1, atol=1e-14, rtol=1e-14)

    ##############################################
    # Matern 2 check

    params["nu"] = np.array([2.0])
    params["lengthscale"] = np.array([1.0])
    Kg = K_cons.K(params, state, np.arange(n)[:, None])
    K2 = evecs_np @ np.diag(np.power(evals_np + 4, -2)) @ evecs_np.T
    np.testing.assert_allclose(Kg, K2, atol=1e-14, rtol=1e-14)

    ##############################################
    # RBF check

    params["nu"] = np.array([np.inf])
    params["lengthscale"] = np.array([1.0])
    Kg = K_cons.K(params, state, np.arange(n)[:, None])
    Ki = evecs_np @ np.diag(np.exp(-0.5 * evals_np)) @ evecs_np.T
    np.testing.assert_allclose(Kg, Ki, atol=1e-14, rtol=1e-14)

    ##############################################
    # Fewer than n eigencomps check

    m = 4
    evecs = graph.get_eigenvectors(m)
    evals = graph.get_eigenvalues(m)
    evals_np, evecs_np = sp.linalg.eigsh(L, m, sigma=1e-8)

    np.testing.assert_allclose(evals[:, 0], evals_np, atol=1e-7, rtol=1e-7)

    np.testing.assert_allclose(np.abs(evecs), np.abs(evecs_np), atol=1e-7, rtol=1e-7)

    K_cons = MaternKarhunenLoeveKernel(graph, m)
    params, state = K_cons.init_params_and_state()
    idx = np.arange(n)[:, None]

    params["nu"] = np.array([np.inf])
    params["lengthscale"] = np.array([1.0])
    Kg = K_cons.K(params, state, idx)
    Ki = evecs_np @ np.diag(np.exp(-0.5 * evals_np)) @ evecs_np.T
    np.testing.assert_allclose(Kg, Ki, atol=1e-7, rtol=1e-7)


def test_graphs_numpy():
    run_tests_with_adj(A)


def test_graphs_scipy_sparse():
    run_tests_with_adj(sp.csr_matrix(A))


# def test_sparse_tensor_ops():

# import torch
# import lab.torch
# import geometric_kernels.torch
# from geometric_kernels.lab_extras import transpose

# mat = torch.sparse_coo_tensor(
#     [[0, 1, 1], [2, 0, 2]], [3, 4, 5], (2, 3))

# np.testing.assert_allclose(
#     transpose(mat).to_dense().numpy(),
#     mat.to_dense().T.numpy()
# )
