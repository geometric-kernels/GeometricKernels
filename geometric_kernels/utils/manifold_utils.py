""" Utilities for dealing with manifolds.  """

import lab as B
import numpy as np

from geometric_kernels.lab_extras import from_numpy


def manifold_laplacian(x: B.Numeric, manifold, egrad, ehess):
    r"""
    Computes the manifold Laplacian of a given function at a given point x.
    The manifold Laplacian equals the trace of the manifold Hessian, i.e.,
    $\Delta_M f(x) = \sum_{i=1}^{d} \nabla^2 f(x_i, x_i)$, where
    $[x_i]_{i=1}^{d}$ is an orthonormal basis of the tangent space at x.

    .. warning::
        This function only works for hyperspheres out of the box. We will
        need to change that in the future.

    .. todo::
        See warning above.

    :param x:
        A point on the manifold at which to compute the Laplacian.
    :param manifold:
        A geomstats manifold.
    :param egrad:
        Euclidean gradient of the given function at x.
    :param ehess:
        Euclidean Hessian of the given function at x.

    :return:
        Manifold Laplacian of the given function at x.

    See :cite:t:`jost2011` (Chapter 3.1) for mathematical details.
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

    .. warning::
        This function only works for hyperspheres out of the box. We will
        need to change that in the future.

    .. todo::
        See warning above.

    :param manifold:
        A geomstats manifold.
    :param x:
        A point on the manifold.

    :return:
        An [d, d]-shaped array containing the orthonormal basis
        on `manifold` at `x`.
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
