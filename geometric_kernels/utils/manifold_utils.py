""" Utilities for dealing with manifolds.  """

import lab as B
import numpy as np
from beartype.typing import Optional

from geometric_kernels.lab_extras import from_numpy


def minkowski_inner_product(vector_a: B.Numeric, vector_b: B.Numeric) -> B.Numeric:
    r"""
    Computes the Minkowski inner product of vectors.

    .. math:: \langle a, b \rangle = a_0 b_0 - a_1 b_1 - \ldots - a_n b_n.

    :param vector_a:
        An [..., n+1]-shaped array of points in the hyperbolic space $\mathbb{H}_n$.
    :param vector_b:
        An [..., n+1]-shaped array of points in the hyperbolic space $\mathbb{H}_n$.

    :return:
        An [...,]-shaped array of inner products.
    """
    assert vector_a.shape == vector_b.shape
    n = vector_a.shape[-1] - 1
    assert n > 0
    diagonal = from_numpy(vector_a, [-1.0] + [1.0] * n)  # (n+1)
    diagonal = B.cast(B.dtype(vector_a), diagonal)
    return B.einsum("...i,...i->...", diagonal * vector_a, vector_b)


def hyperbolic_distance(
    x1: B.Numeric, x2: B.Numeric, diag: Optional[bool] = False
) -> B.Numeric:
    """
    Compute the hyperbolic distance between `x1` and `x2`.

    The code is a reimplementation of
    `geomstats.geometry.hyperboloid.HyperbolicMetric` for `lab`.

    :param x1:
        An [N, n+1]-shaped array of points in the hyperbolic space.
    :param x2:
        An [M, n+1]-shaped array of points in the hyperbolic space.
    :param diag:
        If True, compute elementwise distance. Requires N = M.

        Default False.

    :return:
        An [N, M]-shaped array if diag=False or [N,]-shaped array
        if diag=True.
    """
    if diag:
        # Compute a pointwise distance between `x1` and `x2`
        x1_ = x1
        x2_ = x2
    else:
        if B.rank(x1) == 1:
            x1 = B.expand_dims(x1)
        if B.rank(x2) == 1:
            x2 = B.expand_dims(x2)

        # compute pairwise distance between arrays of points `x1` and `x2`
        # `x1` (N, n+1)
        # `x2` (M, n+1)
        x1_ = B.tile(x1[..., None, :], 1, x2.shape[0], 1)  # (N, M, n+1)
        x2_ = B.tile(x2[None], x1.shape[0], 1, 1)  # (N, M, n+1)

    sq_norm_1 = minkowski_inner_product(x1_, x1_)
    sq_norm_2 = minkowski_inner_product(x2_, x2_)
    inner_prod = minkowski_inner_product(x1_, x2_)

    cosh_angle = -inner_prod / B.sqrt(sq_norm_1 * sq_norm_2)

    one = B.cast(B.dtype(cosh_angle), from_numpy(cosh_angle, [1.0]))
    large_constant = B.cast(B.dtype(cosh_angle), from_numpy(cosh_angle, [1e24]))

    # clip values into [1.0, 1e24]
    cosh_angle = B.where(cosh_angle < one, one, cosh_angle)
    cosh_angle = B.where(cosh_angle > large_constant, large_constant, cosh_angle)

    dist = B.log(cosh_angle + B.sqrt(cosh_angle**2 - 1))  # arccosh
    dist = B.cast(B.dtype(x1_), dist)
    return dist


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
