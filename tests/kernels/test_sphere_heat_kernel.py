import numpy as np
import torch
import lab as B

import geometric_kernels.torch
from geometric_kernels.spaces.hypersphere import Hypersphere
from geometric_kernels.kernels.geometric_kernels import MaternKarhunenLoeveKernel


_TRUNCATION_LEVEL = 10
_NU = 2.5


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

    projected_onb = np.ones_like(ambient_onb)

    for j in range(ambient_dim):
        cur_vec = ambient_onb[j, :]
        projected_onb[j, :] = manifold.to_tangent(cur_vec, base_point=x)

    projected_onb_eigvals, projected_onb_eigvecs = np.linalg.eigh(projected_onb)

    # Getting rid of the zero eigenvalues:
    projected_onb_eigvals = projected_onb_eigvals[ambient_dim - manifold_dim:]
    projected_onb_eigvecs = projected_onb_eigvecs[:, ambient_dim - manifold_dim:]

    assert(np.all(np.isclose(projected_onb_eigvals, 1.)))

    return projected_onb_eigvecs


def test_sphere_heat_kernel():
    # Parameters
    grid_size = 4
    nb_samples = 10
    dimension = 3

    # Create manifold
    hypersphere = Hypersphere(dim=dimension)

    # Generate samples
    ts = torch.linspace(0.1, 1, grid_size, requires_grad=True)
    xs = torch.tensor(np.array(hypersphere.random_point(nb_samples)), requires_grad=True)
    ys = xs

    # Define kernel
    kernel = MaternKarhunenLoeveKernel(hypersphere, _TRUNCATION_LEVEL)
    params, state = kernel.init_params_and_state()
    params["nu"] = torch.tensor(torch.inf)

    # Define heat kernel function
    def heat_kernel(t, x, y):
        params["lengthscale"] = B.sqrt(2*t)
        return kernel.K(params, state, x, y)

    for t in ts:
        for x in xs:
            for y in ys:
                # Compute the derivative of the kernel function wrt t
                dfdt, _, _ = torch.autograd.grad(heat_kernel(t, x[None], y[None]), (t, x, y))
                # Compute the Laplacian of the kernel on the manifold
                egrad = lambda u: torch.autograd.grad(heat_kernel(t, u[None], y[None]), (t, u, y))[1]  # pylint: disable=E731
                fx = lambda u: heat_kernel(t, u[None], y[None])  # pylint: disable=E731
                ehess = lambda u, h: torch.autograd.functional.hvp(fx, u, h)[1]  # pylint: disable=E731
                lapf = manifold_laplacian(hypersphere, x, egrad, ehess)

                # Check that they match
                assert np.isclose(dfdt.detach().numpy(), lapf)
