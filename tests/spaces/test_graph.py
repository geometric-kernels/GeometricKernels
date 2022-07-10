
import numpy as np
from scipy.sparse.linalg import eigsh
from geometric_kernels.spaces import Graph
from geometric_kernels.kernels import MaternKarhunenLoeveKernel

A = np.array([
    [0, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0],
]).astype('float')

def test_graphs():
    # Inits
    n = len(A)
    graph = Graph(A)

    # Laplacian
    L = (np.diag(A.sum(axis=0)) - A)
    assert (graph._laplacian == L).all(), 'Laplacian does not match.'

    # Eigendecomposition
    u = graph.get_eigenvectors(n)
    l = graph.get_eigenvalues(n)
    ln, un = np.linalg.eigh(L)
    np.testing.assert_allclose(l[:, 0], ln, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(np.abs(u), np.abs(un), atol=1e-14, rtol=1e-14)
    graph.get_eigenfunctions(n) # doesn't fail

    # Kernel
    K_cons = MaternKarhunenLoeveKernel(graph, n)
    params, state = K_cons.init_params_and_state()
    idx = np.arange(n)[:, None]

    ## Matern 1
    params["nu"] = np.array([1.0])
    params["lengthscale"] = np.array([1.])
    Kg = K_cons.K(params, state, idx)
    K1 = un @ np.diag(np.power(ln + 2, -1)) @ un.T
    np.testing.assert_allclose(Kg, K1, atol=1e-14, rtol=1e-14)

    ## Matern 2
    params["nu"] = np.array([2.0])
    params["lengthscale"] = np.array([1.])
    Kg = K_cons.K(params, state, np.arange(n)[:, None])
    K2 = un @ np.diag(np.power(ln + 4, -2)) @ un.T
    np.testing.assert_allclose(Kg, K2, atol=1e-14, rtol=1e-14)

    ## Matern inf
    params["nu"] = np.array([np.inf])
    params["lengthscale"] = np.array([1.])
    Kg = K_cons.K(params, state, np.arange(n)[:, None])
    Ki = un @ np.diag(np.exp(-0.5*ln)) @ un.T
    np.testing.assert_allclose(Kg, Ki, atol=1e-14, rtol=1e-14)

    # Kernel with fewer eigencomps
    m = 4
    u = graph.get_eigenvectors(m)
    l = graph.get_eigenvalues(m)
    ln, un = eigsh(L, m, sigma=1e-8)
    np.testing.assert_allclose(l[:, 0], ln, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(np.abs(u), np.abs(un), atol=1e-7, rtol=1e-7)

    K_cons = MaternKarhunenLoeveKernel(graph, m)
    params, state = K_cons.init_params_and_state()
    state['eigenvalues_laplacian'][0, 0] = ln[0] = 1e-15
    idx = np.arange(n)[:, None]

    params["nu"] = np.array([np.inf])
    params["lengthscale"] = np.array([1.])
    Kg = K_cons.K(params, state, idx)
    Ki = un @ np.diag(np.exp(-0.5*ln)) @ un.T
    np.testing.assert_allclose(Kg, Ki, atol=1e-7, rtol=1e-7)

