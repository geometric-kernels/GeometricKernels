"""
Sampling from the Gaussian and Student-t probability densities,
backend-agnostic.
"""
import operator
from functools import reduce

import lab as B
import numpy as np
from sympy import Poly, Product, symbols

from geometric_kernels.lab_extras import dtype_double


def student_t_sample(key, size, deg_freedom, dtype=None):
    r"""
    Sample from the Student-t distribution with `deg_freedom` degrees of freedom,
    using `key` random state, returning sample of the shape `size`.

    Student-t random variable with `nu` degrees of freedom can be represented as
    :math:`T=\frac{Z}{\sqrt{V/\nu}}`, where `Z` is the standard normal r.v. and
    `V` is :math:`\chi^2(\nu)` r.v. The :math:`\chi^2(\nu)` distribution is the
    same as `\Gamma(\nu / 2, 2)` distribution, and therefore
    :math:`V/\nu \sim \Gamma(\nu / 2, 2 / \nu)`

    We use these properties to sample the student-t random variable.

    :param key: either `np.random.RandomState`, `tf.random.Generator`,
                `torch.Generator` or `jax.tensor` (representing random state).
    :param size: shape of the returned sample.
    :param deg_freedom: degrees of freedom of the student-t distribution.
    :param dtype: dtype of the returned tensor.
    """
    dtype = dtype or dtype_double(key)
    key, z = B.randn(key, dtype, *size)
    key, g = B.randgamma(
        key,
        dtype,
        *size,
        alpha=deg_freedom / 2,
        scale=2 / deg_freedom,
    )

    u = z / B.sqrt(g)
    return key, u


def base_density_sample(key, size, params, dim):
    r"""
    The Matern kernel's spectral density is of the form
    :math:`p_{\nu,\kappa}(\lambda)`,
    where :math:`\nu` is the smoothness parameter, :math:`\kappa`
    is the lengthscale and :math:`p_{\nu,\kappa}` is the Student-t
    or Normal density, depending on the smoothness.

    We call it "base density" and this function returns a sample from it.

    :param key: either `np.random.RandomState`, `tf.random.Generator`,
                `torch.Generator` or `jax.tensor` (representing random state).
    :param size: shape of the returned sample.
    :param params: params of the kernel.
    :param dim: dimensionality of the space the kernel is defined on.
    """
    assert "nu" in params
    assert "lengthscale" in params

    nu = params["nu"]
    L = params["lengthscale"]

    if nu == np.inf:
        # sample from Gaussian
        key, u = B.randn(key, dtype_double(key), *size)
    elif nu > 0:
        # sample from the student-t with 2\nu + dim - 1 degrees of freedom
        deg_freedom = 2 * nu + dim - 1
        key, u = student_t_sample(key, size, deg_freedom)

    return key, u / L


def alphas(n):
    r"""
    Compute alphas for Prop. 16 & 17 for the hyperbolic space of dimension `n`.

    :param n: dimension of the hyperbolic space, n >= 2.

    TODO: precompute these, rather than computing in runtime.
    """
    assert n >= 2
    x, j = symbols("x, j")
    if (n % 2) == 0:
        m = n // 2
        prod = x * Product(x**2 + (2 * j - 3) ** 2 / 4, (j, 2, m)).doit()
    else:
        m = (n - 1) // 2
        prod = Product(x**2 + j**2, (j, 0, m - 1)).doit()
    return np.array(Poly(prod, x).all_coeffs()).astype(np.float64)[::-1]


def sample_mixture_heat(key, alpha, lengthscale):
    r"""
    Sample from the mixture distribution from Prop. 16 for specific alphas
    `alpha` and length scale (kappa) `lengthscale` using `key` random state.

    :param key: either `np.random.RandomState`, `tf.random.Generator`,
                `torch.Generator` or `jax.tensor` (representing random state).
    :param alpha: unnormalized coefficients of the mixture.
    :param lengthscale: length scale (kappa).

    TODO: reparameterization trick.
    """
    assert B.rank(alpha) == 1
    m = B.shape(alpha)[0] - 1
    assert m >= 0
    dtype = B.dtype(lengthscale)
    js = B.range(dtype, 0, m + 1)

    # Gamma((js+1)/2) should be positive real, so G = exp(log(abs(G)))
    beta = 2 ** ((1 - js) / 2) / B.exp(B.loggamma((js + 1) / 2)) * lengthscale
    cs_unnorm = alpha / beta
    cs = cs_unnorm / B.sum(cs_unnorm)
    key, ind = B.choice(key, js, 1, p=cs)

    # Gamma(2nu, 2) distribution is the same as chi2(nu) distribution
    key, s = B.randgamma(key, dtype, 1, alpha=(ind + 1) / 2, scale=2)
    s = B.sqrt(s) / lengthscale
    return key, s


def sample_mixture_matern(key, alpha, lengthscale, nu, dim):
    r"""
    Sample from the mixture distribution from Prop. 17 for specific alphas
    `alpha`, length scale (kappa) `lengthscale`, smoothness `nu` and dimnesion
    `dim`, using `key` random state.

    :param key: either `np.random.RandomState`, `tf.random.Generator`,
                `torch.Generator` or `jax.tensor` (representing random state).
    :param alpha: unnormalized coefficients of the mixture.
    :param lengthscale: length scale (kappa).
    :param nu: smoothness parameter of Matern kernels.
    :param dim: dimension of the hyperbolic space.

    TODO: reparameterization trick.
    """
    assert B.rank(alpha) == 1
    m = B.shape(alpha)[0] - 1
    assert m >= 0
    dtype = B.dtype(lengthscale)
    js = B.range(dtype, 0, m + 1)
    gamma = 2 * nu / lengthscale**2 + ((dim - 1) / 2) ** 2

    # B(x, y) = Gamma(x) Gamma(y) / Gamma(x+y)
    beta = B.exp(
        B.loggamma((js + 1) / 2)
        + B.loggamma(nu + (dim - js - 1) / 2)
        - B.loggamma(nu + dim / 2)
    )
    beta = 2.0 / beta
    beta = beta * B.power(gamma, nu + (dim - js - 1) / 2)

    cs_unnorm = alpha / beta
    cs = cs_unnorm / B.sum(cs_unnorm)

    key, ind = B.choice(key, js, 1, p=cs)
    key, p = B.randbeta(
        key, dtype, 1, alpha=(ind + 1) / 2, beta=nu + (dim - ind - 1) / 2
    )
    p = p / (1 - p)  # beta prime distribution
    s = B.sqrt(gamma * p)
    return key, s


def hyperbolic_density_sample(key, size, params, dim):
    r"""
    :param key: either `np.random.RandomState`, `tf.random.Generator`,
                `torch.Generator` or `jax.tensor` (representing random state).
    :param size: shape of the returned sample.
    :param params: params of the kernel.
    :param dim: dimensionality of the space the kernel is defined on.
    """
    assert "nu" in params
    assert "lengthscale" in params

    nu = params["nu"]
    L = params["lengthscale"]

    alpha = alphas(dim)

    def base_sampler(key):
        if nu == np.inf:
            return sample_mixture_heat(key, alpha, L)
        else:
            return sample_mixture_matern(key, alpha, L, nu, dim)

    samples = []
    while len(samples) < reduce(operator.mul, size, 1):
        key, proposal = base_sampler(key)

        # accept with probability tanh(pi*proposal)
        key, u = B.rand(key, dtype_double(key), 1)
        acceptance = B.all(u < B.tanh(B.pi * proposal))

        key, sign_z = B.rand(key, dtype_double(key), 1)
        sign = B.sign(sign_z - 0.5)  # +1 or -1 with probability 0.5
        if ((dim % 2) == 1) or acceptance:
            samples.append(sign * proposal)
    samples = B.reshape(B.concat(*samples), *size)
    return key, B.cast(dtype_double(key), samples)
