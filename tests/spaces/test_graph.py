import warnings

import lab as B
import numpy as np
import pytest

from geometric_kernels.jax import *  # noqa
from geometric_kernels.kernels import MaternGeometricKernel
from geometric_kernels.spaces import Graph
from geometric_kernels.tensorflow import *  # noqa
from geometric_kernels.torch import *  # noqa

from ..data import (
    TEST_GRAPH_ADJACENCY,
    TEST_GRAPH_LAPLACIAN,
    TEST_GRAPH_LAPLACIAN_NORMALIZED,
)
from ..helper import check_function_with_backend, np_to_backend

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

A = TEST_GRAPH_ADJACENCY
L = TEST_GRAPH_LAPLACIAN


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize(
    "backend", ["numpy", "tensorflow", "torch", "jax", "scipy_sparse"]
)
def test_laplacian(normalized, backend):

    # Check that the Laplacian is computed correctly.
    check_function_with_backend(
        backend,
        TEST_GRAPH_LAPLACIAN if not normalized else TEST_GRAPH_LAPLACIAN_NORMALIZED,
        lambda adj: Graph(adj, normalize_laplacian=normalized)._laplacian,
        TEST_GRAPH_ADJACENCY,
    )


@pytest.mark.parametrize(
    "L", [TEST_GRAPH_ADJACENCY.shape[0], TEST_GRAPH_ADJACENCY.shape[0] // 2]
)
@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize(
    "backend", ["numpy", "tensorflow", "torch", "jax", "scipy_sparse"]
)
def test_eigendecomposition(L, normalized, backend):
    laplacian = np_to_backend(
        TEST_GRAPH_LAPLACIAN if not normalized else TEST_GRAPH_LAPLACIAN_NORMALIZED,
        backend,
    )

    def eigendiff(adj):
        graph = Graph(adj, normalize_laplacian=normalized)

        eigenvalue_mat = B.diag_construct(graph.get_eigenvalues(L)[:, 0])
        eigenvectors = graph.get_eigenvectors(L)
        # If the backend is scipy_sparse, convert eigenvalues/eigenvectors,
        # which are always supposed to be dense arrays, to sparse arrays.
        if backend == "scipy_sparse":
            import scipy.sparse as sp

            eigenvalue_mat = sp.csr_array(eigenvalue_mat)
            eigenvectors = sp.csr_array(eigenvectors)

        laplace_x_eigvecs = laplacian @ eigenvectors
        eigvals_x_eigvecs = eigenvectors @ eigenvalue_mat
        return laplace_x_eigvecs - eigvals_x_eigvecs

    check_function_with_backend(
        backend,
        np.zeros((TEST_GRAPH_ADJACENCY.shape[0], L)),
        eigendiff,
        TEST_GRAPH_ADJACENCY,
    )


@pytest.mark.parametrize("nu, lengthscale", [(1.0, 1.0), (2.0, 1.0), (np.inf, 1.0)])
@pytest.mark.parametrize("sparse_adj", [True, False])
@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize(
    "backend", ["numpy", "tensorflow", "torch", "jax"]
)  # The kernels never take sparse parameters and never output sparse matrices, thus we don't test scipy_sparse. The fact that the adjacency matrix may be sparse is tested when sparse_adj is True.
def test_matern_kernels(nu, lengthscale, sparse_adj, normalized, backend):

    laplacian = (
        TEST_GRAPH_LAPLACIAN if not normalized else TEST_GRAPH_LAPLACIAN_NORMALIZED
    )

    evals_np, evecs_np = np.linalg.eigh(laplacian)
    evecs_np *= np.sqrt(laplacian.shape[0])

    def evaluate_kernel(adj, nu, lengthscale):
        dtype = B.dtype(adj)
        if sparse_adj:
            adj = np_to_backend(B.to_numpy(adj), "scipy_sparse")
        graph = Graph(adj, normalize_laplacian=normalized)
        kernel = MaternGeometricKernel(graph)
        return kernel.K(
            {"nu": nu, "lengthscale": lengthscale},
            B.range(dtype, adj.shape[0])[:, None],
        )

    if nu < np.inf:
        K = (
            evecs_np
            @ np.diag(np.power(evals_np + 2 * nu / lengthscale**2, -nu))
            @ evecs_np.T
        )
    else:
        K = evecs_np @ np.diag(np.exp(-(lengthscale**2) / 2 * evals_np)) @ evecs_np.T
    K = K / np.mean(K.diagonal())

    check_function_with_backend(
        backend,
        K,
        evaluate_kernel,
        TEST_GRAPH_ADJACENCY,
        np.array([nu]),
        np.array([lengthscale]),
    )
