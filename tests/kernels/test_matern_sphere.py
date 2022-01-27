import matplotlib.pyplot as plt
import math
import numpy as rnp
import autograd.numpy as np
from autograd import grad, hessian_vector_product

from geometric_kernels.kernels.geometric_kernels import MaternIntegratedKernel
from geometric_kernels.spaces.sphere import Sphere

_NUM_POINTS = 50
_NU = 2.5


def plot_distance_vs_kernel_sphere():
    hypersphere = Sphere(dim=2)
    lengthscale = 1.

    # Create a set of points along a geodesic
    base = rnp.r_[1., 0., 0.]
    point = rnp.r_[-0.9, 0.1, 0.]
    point = point / rnp.linalg.norm(point)

    geodesic = hypersphere.metric.geodesic(initial_point=base, end_point=point)
    x1 = geodesic(rnp.linspace(0., 1., 100))
    print('geodesic', x1.shape)

    x2 = x1[-1, None]

    # Compute sphere distance
    distances = hypersphere.distance(x1, x2)

    # Compute heat and Matérn kernels
    heat_kernel_vals = hypersphere.heat_kernel(distances, rnp.array(0.5*lengthscale**2)[None])  # Lengthscale to heat kernel t parameter
    heat_kernel_vals_normalized = heat_kernel_vals / heat_kernel_vals[-1]
    matern_kernel = MaternIntegratedKernel(hypersphere, _NU, _NUM_POINTS)
    matern_kernel_vals = matern_kernel.K(x1, x2, lengthscale=1.)

    # Plot kernel value in function of the distance
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    plt.plot(distances, heat_kernel_vals_normalized, color='gold', linewidth=3)
    plt.plot(distances, matern_kernel_vals, color='red', linewidth=1)
    ax.tick_params(labelsize=22)
    ax.set_xlabel(r'distance', fontsize=30)
    ax.set_ylabel(r'k', fontsize=30)
    ax.legend([r'SE', r'Matern'], fontsize=24)
    plt.show()


def gegenbauer_polynomial_np(n, alpha, z):
    """
    This function computes the Gegenbauer polynomial C_n^alpha(z).
    Parameters
    ----------
    :param n: Gegenbauer polynomial parameter n
    :param alpha: Gegenbauer polynomial parameter alpha
    :param z: Gegenbauer polynomial function input
    Returns
    -------
    :return: Gegenbauer polynomial C_n^alpha(z)
    """
    # The general formula down below does not work for alpha = 0. Because of
    # that, this case is considered separately.
    if np.isclose(alpha, 0):
        # This should basically return np.cos(n*np.arccos(z)), the problem is that
        # autograd is having troubles with differentiating this expression at z=0.
        # Because of this, we use the relation cos(n*x) = T_n(cos(x)) where T_n is
        # the Chebyshev polynomial of the first kind.
        poly_index = np.zeros(n + 1)
        poly_index[-1] = 1.
        coeffs = rnp.polynomial.chebyshev.cheb2poly(poly_index)
        polynomial = 0.
        for j, coeff in enumerate(coeffs):
            polynomial += coeff*np.power(z, j)
        return polynomial

    # Initialization
    polynomial = 0.
    gamma_alpha = math.gamma(alpha)
    # Computes the summation series
    for i in range(math.floor(n/2)+1):
        polynomial += math.pow(-1, i) * np.power(2*z, n-2*i) \
                      * (math.gamma(n-i+alpha) / (gamma_alpha * math.factorial(i) * math.factorial(n-2*i)))
    return polynomial


def sphere_ehess2rhess(manifold, x, egrad, ehess, direction):
    normal_gradient = egrad - manifold.to_tangent(egrad, base_point=x)
    return manifold.to_tangent(ehess, base_point=x) - manifold.metric.inner_product(x, normal_gradient, base_point=x) * direction


def lap(manifold, egrad, ehess, x):
    dim = manifold.dim
    onb = tangent_onb(manifold, x)
    result = 0.
    for j in range(dim):
        cur_vec = onb[:, j]
        hess_vec_prod = sphere_ehess2rhess(manifold, x, egrad(x), ehess(x, cur_vec), cur_vec)
        result += manifold.metric.inner_product(hess_vec_prod, cur_vec, base_point=x)

    return result


def tangent_onb(manifold, x, verbose=False):
    ambient_dim = manifold.dim + 1
    manifold_dim = manifold.dim
    ambient_onb = np.eye(ambient_dim)

    if verbose:
        print('Ambient onb:')
        print(ambient_onb)

    projected_onb = np.ones_like(ambient_onb)

    for j in range(ambient_dim):
        cur_vec = ambient_onb[j, :]
        projected_onb[j, :] = manifold.to_tangent(cur_vec, base_point=x)

    if verbose:
        print('Projected onb')
        print(projected_onb)

    projected_onb_eigvals, projected_onb_eigvecs = np.linalg.eigh(projected_onb)

    # Getting rid of the zero eigenvalues:
    projected_onb_eigvals = projected_onb_eigvals[ambient_dim - manifold_dim:]
    projected_onb_eigvecs = projected_onb_eigvecs[:, ambient_dim - manifold_dim:]

    if verbose:
        print('Eigenvalues:')
        print(projected_onb_eigvals)
        print('Eigenvectors:')
        print(projected_onb_eigvecs)
    assert(np.all(np.isclose(projected_onb_eigvals, 1.)))

    return projected_onb_eigvecs


def test_heat_equation():
    # Parameters
    grid_size = 4
    nb_samples = 10
    dimension = 3
    N = 2
    lamb = N * (N + dimension - 1)

    # Create manifold
    hypersphere = Sphere(dim=dimension)

    # Generate samples
    ts = np.linspace(0, 1, grid_size)
    xs = hypersphere.random_point(nb_samples)
    ys = xs

    # Define function
    def f(t, x, y):
        return np.exp(-t * lamb) * gegenbauer_polynomial_np(N, (dimension - 1) / 2, np.sum(np.multiply(x, y), axis=-1))

    for t in ts:
        for x in xs:
            for y in ys:
                dfdt = grad(f, 0)(t, x, y)
                egrad = lambda u: grad(f, 1)(t, u, y)
                ehess = lambda u, h: hessian_vector_product(f, 1)(t, u, y, h)
                lapf = lap(hypersphere, egrad, ehess, x)
                print('t = %0.2f' % t, 'x =', x, 'y =', y)
                print('df/dt(t, x, y)   = %0.8f' % dfdt)
                print('Delta f(t, x, y) = %0.8f' % lapf)
                assert np.isclose(dfdt, lapf)


if __name__ == "__main__":
    test_heat_equation()
    plot_distance_vs_kernel_sphere()
