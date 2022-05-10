import lab as B
import numpy as np

from geometric_kernels.lab_extras import from_numpy


def manifold_laplacian(x: B.Numeric, manifold, egrad, ehess):
    r"""
    Computes the manifold Laplacian of a given function at a given point x.
    The manifold Laplacian equals the trace of the manifold Hessian, i.e.,
    :math:`\Delta_M f(x) = \sum_{i=0}^{D-1} \nabla^2 f(x_i, x_i)`,
    where :math:`[x_i]_{i=0}^{D-1}` is an orthonormal basis of the tangent
    space at x.

    :param x: point on the manifold at which to compute the Laplacian
    :param manifold: manifold space, based on geomstats
    :param egrad: Euclidean gradient of the function
    :param ehess: Euclidean Hessian of the function

    :return: manifold Laplacian

    References:

        [1] J. Jost.
            Riemannian geometry and geometric analysis. Springer, 2017.
            Chapter 3.1.
    """
    dim = manifold.dim

    onb = tangent_onb(manifold, B.to_numpy(x))
    result = 0.0
    for j in range(dim):
        cur_vec = onb[:, j]
        egrad_x = B.to_numpy(egrad(x))
        ehess_x = B.to_numpy(ehess(x, from_numpy(x, cur_vec)))
        hess_vec_prod = manifold.ehess2rhess(B.to_numpy(x), egrad_x, ehess_x, cur_vec)
        result += manifold.metric.inner_product(
            hess_vec_prod, cur_vec, base_point=B.to_numpy(x)
        )

    return result


def tangent_onb(manifold, x):
    r"""
    Computes an orthonormal basis on the tangent space at x.

    :param manifold: manifold space, based on geomstats
    :param x: point on the manifold

    :return: [num, num] array containing the orthonormal basis
    """
    ambient_dim = manifold.dim + 1
    manifold_dim = manifold.dim
    ambient_onb = np.eye(ambient_dim)

    projected_onb = manifold.to_tangent(ambient_onb, base_point=x)

    projected_onb_eigvals, projected_onb_eigvecs = np.linalg.eigh(projected_onb)

    # Getting rid of the zero eigenvalues:
    projected_onb_eigvals = projected_onb_eigvals[ambient_dim - manifold_dim :]
    projected_onb_eigvecs = projected_onb_eigvecs[:, ambient_dim - manifold_dim :]

    assert np.all(np.isclose(projected_onb_eigvals, 1.0))

    return projected_onb_eigvecs
