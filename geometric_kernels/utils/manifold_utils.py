import numpy as np
import torch


def manifold_laplacian(manifold, x, egrad, ehess):
    r"""
    Computes the manifold Laplacian of a given function at a given point x.

    :param manifold: manifold space, based on geomstats
    :param x: point on the manifold at which to compute the Laplacian
    :param egrad: Euclidean gradient of the function
    :param ehess: Euclidean Hessian of the function

    :return: manifold Laplacian
    """
    dim = manifold.dim
    onb = torch.tensor(tangent_onb(manifold, x.detach().numpy()))
    result = 0.
    for j in range(dim):
        cur_vec = onb[:, j]
        egrad_x = egrad(x).detach().numpy()
        ehess_x = ehess(x, cur_vec).detach().numpy()
        hess_vec_prod = manifold.ehess2rhess(x.detach().numpy(), egrad_x, ehess_x, cur_vec.detach().numpy())
        result += manifold.metric.inner_product(hess_vec_prod, cur_vec.detach().numpy(), base_point=x.detach().numpy())

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
    projected_onb_eigvals = projected_onb_eigvals[ambient_dim - manifold_dim:]
    projected_onb_eigvecs = projected_onb_eigvecs[:, ambient_dim - manifold_dim:]

    assert(np.all(np.isclose(projected_onb_eigvals, 1.)))

    return projected_onb_eigvecs
