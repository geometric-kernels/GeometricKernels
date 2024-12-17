import lab as B
import numpy as np
import pytest

from geometric_kernels.jax import *  # noqa
from geometric_kernels.kernels import (
    MaternGeometricKernel,
    MaternHodgeCompositionalKernel,
    MaternKarhunenLoeveKernel,
)
from geometric_kernels.spaces import GraphEdges
from geometric_kernels.tensorflow import *  # noqa
from geometric_kernels.torch import *  # noqa

from ..data import (
    TEST_GRAPH_EDGES_ADJACENCY,
    TEST_GRAPH_EDGES_DOWN_LAPLACIAN,
    TEST_GRAPH_EDGES_NUM_NODES,
    TEST_GRAPH_EDGES_ORIENTED_EDGES,
    TEST_GRAPH_EDGES_ORIENTED_TRIANGLES,
    TEST_GRAPH_EDGES_ORIENTED_TRIANGLES_AUTO,
    TEST_GRAPH_EDGES_TRIANGLES,
    TEST_GRAPH_EDGES_UP_LAPLACIAN,
)
from ..helper import check_function_with_backend, create_random_state, np_to_backend


@pytest.mark.parametrize(
    "triangles, oriented_triangles",
    [
        (None, TEST_GRAPH_EDGES_ORIENTED_TRIANGLES_AUTO),
        (TEST_GRAPH_EDGES_TRIANGLES, TEST_GRAPH_EDGES_ORIENTED_TRIANGLES),
    ],
)
@pytest.mark.parametrize(
    "backend", ["numpy", "tensorflow", "torch", "jax", "scipy_sparse"]
)
def test_from_adjacency(backend, triangles, oriented_triangles):

    type_reference = create_random_state(backend)

    # check that oriented_edges array is correctly computed from adjacency
    B.all(
        TEST_GRAPH_EDGES_ORIENTED_EDGES
        == GraphEdges.from_adjacency(
            TEST_GRAPH_EDGES_ADJACENCY, type_reference, triangles=triangles
        ).oriented_edges
    )

    # check that oriented_triangles array is correctly computed from adjacency
    B.all(
        oriented_triangles
        == GraphEdges.from_adjacency(
            TEST_GRAPH_EDGES_ADJACENCY, type_reference, triangles=triangles
        ).oriented_triangles
    )


@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_laplacian(backend):

    check_function_with_backend(
        backend,
        TEST_GRAPH_EDGES_UP_LAPLACIAN,
        lambda or_e, or_t: GraphEdges(
            TEST_GRAPH_EDGES_NUM_NODES, or_e, or_t
        )._up_laplacian,
        TEST_GRAPH_EDGES_ORIENTED_EDGES,
        TEST_GRAPH_EDGES_ORIENTED_TRIANGLES,
    )

    check_function_with_backend(
        backend,
        TEST_GRAPH_EDGES_DOWN_LAPLACIAN,
        lambda or_e, or_t: GraphEdges(
            TEST_GRAPH_EDGES_NUM_NODES, or_e, or_t
        )._down_laplacian,
        TEST_GRAPH_EDGES_ORIENTED_EDGES,
        TEST_GRAPH_EDGES_ORIENTED_TRIANGLES,
    )

    check_function_with_backend(
        backend,
        TEST_GRAPH_EDGES_UP_LAPLACIAN + TEST_GRAPH_EDGES_DOWN_LAPLACIAN,
        lambda or_e, or_t: GraphEdges(
            TEST_GRAPH_EDGES_NUM_NODES, or_e, or_t
        )._hodge_laplacian,
        TEST_GRAPH_EDGES_ORIENTED_EDGES,
        TEST_GRAPH_EDGES_ORIENTED_TRIANGLES,
    )


@pytest.mark.parametrize(
    "L",
    [
        TEST_GRAPH_EDGES_ORIENTED_EDGES.shape[0],
        TEST_GRAPH_EDGES_ORIENTED_EDGES.shape[0] // 2,
    ],
)
@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_eigendecomposition(L, backend):
    hodge_laplacian = np_to_backend(
        TEST_GRAPH_EDGES_UP_LAPLACIAN + TEST_GRAPH_EDGES_DOWN_LAPLACIAN,
        backend,
    )

    def eigendiff(or_e, or_t):
        graph_edges = GraphEdges(TEST_GRAPH_EDGES_NUM_NODES, or_e, or_t)

        eigenvalue_mat = B.diag_construct(graph_edges.get_eigenvalues(L)[:, 0])
        eigenvectors = graph_edges.get_eigenvectors(L)

        laplace_x_eigvecs = hodge_laplacian @ eigenvectors
        eigvals_x_eigvecs = eigenvectors @ eigenvalue_mat
        return laplace_x_eigvecs - eigvals_x_eigvecs

    check_function_with_backend(
        backend,
        np.zeros((TEST_GRAPH_EDGES_ORIENTED_EDGES.shape[0], L)),
        eigendiff,
        TEST_GRAPH_EDGES_ORIENTED_EDGES,
        TEST_GRAPH_EDGES_ORIENTED_TRIANGLES,
    )


@pytest.mark.parametrize(
    "hodge_type, projection_matrix, projection_matrix_id",
    [
        ("harmonic", TEST_GRAPH_EDGES_UP_LAPLACIAN, "up"),
        ("harmonic", TEST_GRAPH_EDGES_DOWN_LAPLACIAN, "down"),
        ("gradient", TEST_GRAPH_EDGES_DOWN_LAPLACIAN, "down"),
        ("curl", TEST_GRAPH_EDGES_UP_LAPLACIAN, "up"),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
@pytest.mark.parametrize(
    "L",
    [
        TEST_GRAPH_EDGES_ORIENTED_EDGES.shape[0],
        TEST_GRAPH_EDGES_ORIENTED_EDGES.shape[0] // 2,
    ],
)
@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_hodge_decomposition(
    hodge_type, projection_matrix, projection_matrix_id, L, backend
):
    # projection_matrix_id we only use for the ids of the test

    n = GraphEdges(
        TEST_GRAPH_EDGES_NUM_NODES,
        TEST_GRAPH_EDGES_ORIENTED_EDGES,
        TEST_GRAPH_EDGES_ORIENTED_TRIANGLES,
    ).get_number_of_eigenpairs(L, hodge_type=hodge_type)

    def proj(or_e, or_t, proj_mat):
        graph_edges = GraphEdges(TEST_GRAPH_EDGES_NUM_NODES, or_e, or_t)
        proj_mat = np_to_backend(proj_mat, backend)

        eigenvectors = graph_edges.get_eigenvectors(L, hodge_type=hodge_type)

        result = proj_mat @ eigenvectors
        return result

    # Check that the eigenvectors of type `hodge_type` lie in the null space
    # of the projection matrix (are harmonic / divergence-free / curl-free).
    check_function_with_backend(
        backend,
        np.zeros((TEST_GRAPH_EDGES_ORIENTED_EDGES.shape[0], n)),
        proj,
        TEST_GRAPH_EDGES_ORIENTED_EDGES,
        TEST_GRAPH_EDGES_ORIENTED_TRIANGLES,
        projection_matrix,
    )


@pytest.mark.parametrize("nu, lengthscale", [(1.0, 1.0), (2.0, 1.0), (np.inf, 1.0)])
@pytest.mark.parametrize("sparse_adj", [True, False])
@pytest.mark.parametrize("hodge_compositional", [True, False])
@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_matern_kernels(nu, lengthscale, hodge_compositional, sparse_adj, backend):

    hodge_laplacian = TEST_GRAPH_EDGES_UP_LAPLACIAN + TEST_GRAPH_EDGES_DOWN_LAPLACIAN
    num_edges = hodge_laplacian.shape[0]

    evals_np, evecs_np = np.linalg.eigh(hodge_laplacian)
    evecs_np *= np.sqrt(hodge_laplacian.shape[0])

    type_reference = create_random_state(backend)

    def evaluate_kernel(nu, lengthscale, xs):
        adj = TEST_GRAPH_EDGES_ADJACENCY
        if sparse_adj:
            adj = np_to_backend(B.to_numpy(TEST_GRAPH_EDGES_ADJACENCY), "scipy_sparse")
        graph_edges = GraphEdges.from_adjacency(
            adj, type_reference, triangles=TEST_GRAPH_EDGES_TRIANGLES
        )

        if hodge_compositional:
            kernel = MaternHodgeCompositionalKernel(graph_edges, num_levels=num_edges)

            # We want MaternHodgeCompositionalKernel to coincide with
            # MaternKarhunenLoeveKernel in this case. For this, we need to
            # set the right coefficients for the three hodge types.

            a = B.reshape(
                B.sum(
                    MaternKarhunenLoeveKernel.spectrum(
                        graph_edges.get_eigenvalues(num_edges, hodge_type="harmonic"),
                        nu,
                        lengthscale,
                        0,
                    )
                ),
                1,
            )

            b = B.reshape(
                B.sum(
                    MaternKarhunenLoeveKernel.spectrum(
                        graph_edges.get_eigenvalues(num_edges, hodge_type="gradient"),
                        nu,
                        lengthscale,
                        0,
                    )
                ),
                1,
            )

            c = B.reshape(
                B.sum(
                    MaternKarhunenLoeveKernel.spectrum(
                        graph_edges.get_eigenvalues(num_edges, hodge_type="curl"),
                        nu,
                        lengthscale,
                        0,
                    )
                ),
                1,
            )

            params = {
                "harmonic": {"logit": a, "nu": nu, "lengthscale": lengthscale},
                "gradient": {"logit": b, "nu": nu, "lengthscale": lengthscale},
                "curl": {"logit": c, "nu": nu, "lengthscale": lengthscale},
            }
        else:
            kernel = MaternKarhunenLoeveKernel(graph_edges, num_levels=num_edges)
            params = {"nu": nu, "lengthscale": lengthscale}

        return kernel.K(params, xs)

    if nu < np.inf:
        K = (
            evecs_np
            @ np.diag(np.power(evals_np + 2 * nu / lengthscale**2, -nu))
            @ evecs_np.T
        )
    else:
        K = evecs_np @ np.diag(np.exp(-(lengthscale**2) / 2 * evals_np)) @ evecs_np.T
    K = K / np.mean(K.diagonal())

    # Check that the kernel matrix is correctly computed. For the Hodge
    # compositional kernel, we only check the case when the hyperparameters
    # make it coincide with the Karhunen-Loève kernel.
    check_function_with_backend(
        backend,
        K,
        evaluate_kernel,
        np.array([nu]),
        np.array([lengthscale]),
        np.arange(1, num_edges + 1)[:, None],
    )


@pytest.mark.parametrize(
    "coeffs, projection_matrix, projection_matrix_id",
    [
        ([1.0, 0.0, 0.0], TEST_GRAPH_EDGES_UP_LAPLACIAN, "up"),
        ([1.0, 0.0, 0.0], TEST_GRAPH_EDGES_DOWN_LAPLACIAN, "down"),
        ([0.0, 1.0, 0.0], TEST_GRAPH_EDGES_DOWN_LAPLACIAN, "down"),
        ([0.0, 0.0, 1.0], TEST_GRAPH_EDGES_UP_LAPLACIAN, "up"),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
@pytest.mark.parametrize("nu, lengthscale", [(1.0, 1.0), (2.0, 1.0), (np.inf, 1.0)])
@pytest.mark.parametrize("backend", ["numpy", "tensorflow", "torch", "jax"])
def test_kernels_hodge_type(
    coeffs, projection_matrix, projection_matrix_id, nu, lengthscale, backend
):
    # projection_matrix_id we only use for the ids of the test

    def proj_kernel(
        or_e,
        or_t,
        coeff_harmonic,
        coeff_gradient,
        coeff_curl,
        nu,
        lengthscale,
        xs,
        projection_matrix,
    ):
        graph_edges = GraphEdges(TEST_GRAPH_EDGES_NUM_NODES, or_e, or_t)

        kernel = MaternGeometricKernel(graph_edges)

        params = {
            "harmonic": {"logit": coeff_harmonic, "nu": nu, "lengthscale": lengthscale},
            "gradient": {"logit": coeff_gradient, "nu": nu, "lengthscale": lengthscale},
            "curl": {"logit": coeff_curl, "nu": nu, "lengthscale": lengthscale},
        }

        return projection_matrix @ kernel.K(params, xs)

    num_edges = TEST_GRAPH_EDGES_ORIENTED_EDGES.shape[0]

    # Check that the image of the Hodge compositional kernel lies in the null
    # space of the projection matrix. Makes sure that buy setting the
    # coefficients of the other two types to zero, the kernel is harmonic /
    # divergence-free / curl-free.
    check_function_with_backend(
        backend,
        np.zeros((num_edges, num_edges)),
        proj_kernel,
        TEST_GRAPH_EDGES_ORIENTED_EDGES,
        TEST_GRAPH_EDGES_ORIENTED_TRIANGLES,
        np.array([coeffs[0]]),
        np.array([coeffs[1]]),
        np.array([coeffs[2]]),
        np.array([nu]),
        np.array([lengthscale]),
        np.arange(1, num_edges + 1)[:, None],
        projection_matrix,
    )
