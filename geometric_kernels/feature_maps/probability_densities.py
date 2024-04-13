"""
This module provide the routines for sampling from the Gaussian and Student-t
probability densities in a backend-agnostic way. It also provides the routines
for sampling the non-standard probability densities that arise in relation to
the :class:`Hyperbolic` and :class:`SymmetricPositiveDefiniteMatrices` spaces.
"""

import operator
from functools import reduce

import lab as B
import numpy as np
from sympy import Poly, Product, symbols

from geometric_kernels.lab_extras import (
    cumsum,
    dtype_double,
    dtype_integer,
    eigvalsh,
    from_numpy,
)
from geometric_kernels.utils.utils import ordered_pairwise_differences


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
    assert B.shape(deg_freedom) == (1,), "deg_freedom must be a 1-vector."
    dtype = dtype or dtype_double(key)
    key, z = B.randn(key, dtype, *size)

    key, g = B.randgamma(
        key,
        dtype,
        *size,
        alpha=deg_freedom / 2,
        scale=2 / deg_freedom,
    )
    g = B.squeeze(g, axis=-1)

    u = z / B.sqrt(g)
    return key, u


def base_density_sample(key, size, params, dim, rho):
    r"""
    The Matérn kernel's spectral density is of the form
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
    :param rho: `rho` vector of the space.
    """
    assert "nu" in params
    assert "lengthscale" in params

    nu = params["nu"]
    L = params["lengthscale"]

    # Note: 1.0 in safe_nu can be replaced by any finite positive value
    safe_nu = B.where(nu == np.inf, B.cast(B.dtype(L), np.r_[1.0]), nu)

    # for nu == np.inf
    # sample from Gaussian
    key, u_nu_infinite = B.randn(key, B.dtype(L), *size)
    # for nu < np.inf
    # sample from the student-t with 2\nu + dim(space) - dim(rho)  degrees of freedom
    deg_freedom = 2 * safe_nu + dim - B.rank(rho)
    key, u_nu_finite = student_t_sample(key, size, deg_freedom, B.dtype(L))

    u = B.where(nu == np.inf, u_nu_infinite, u_nu_finite)

    scale_nu_infinite = L
    scale_nu_finite = B.sqrt(deg_freedom) / B.sqrt(2 * safe_nu / L**2 + B.sum(rho**2))

    scale = B.where(nu == np.inf, scale_nu_infinite, scale_nu_finite)

    scale = B.cast(B.dtype(u), scale)
    return key, u / scale


def randcat_fix(key, dtype, size, p):
    """
    Sample from the categorical variable with probabilities `p`.
    """
    p = p / B.sum(p, axis=-1, squeeze=False)
    # Perform sampling routine.
    cdf = cumsum(p, axis=-1)
    key, u = B.rand(key, dtype, size, *B.shape(p)[:-1])
    inds = B.argmax(B.cast(dtype_integer(key), u[..., None] < cdf[None]), axis=-1)
    return key, B.cast(dtype, inds)


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

    alpha = B.cast(dtype, from_numpy(beta, alpha))
    cs_unnorm = alpha / beta
    cs = cs_unnorm / B.sum(cs_unnorm)
    key, ind = randcat_fix(key, dtype, 1, cs)

    # Gamma(nu/2, 2) distribution is the same as chi2(nu) distribution
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
    :param nu: smoothness parameter of Matérn kernels.
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

    alpha = B.cast(dtype, from_numpy(beta, alpha))
    cs_unnorm = alpha / beta
    cs = cs_unnorm / B.sum(cs_unnorm)

    key, ind = randcat_fix(key, dtype, 1, cs)
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
        # Note: 1.0 in safe_nu can be replaced by any finite positive value
        safe_nu = B.where(nu == np.inf, B.cast(B.dtype(L), np.r_[1.0]), nu)

        # for nu == np.inf
        key, sample_mixture_nu_infinite = sample_mixture_heat(key, alpha, L)
        # for nu < np.inf
        key, sample_mixture_nu_finite = sample_mixture_matern(
            key, alpha, L, safe_nu, dim
        )

        return key, B.where(
            nu == np.inf, sample_mixture_nu_infinite, sample_mixture_nu_finite
        )

    samples = []
    while len(samples) < reduce(operator.mul, size, 1):
        key, proposal = base_sampler(key)

        # accept with probability tanh(pi*proposal)
        key, u = B.rand(key, B.dtype(L), 1)
        acceptance = B.all(u < B.tanh(B.pi * proposal))

        key, sign_z = B.rand(key, B.dtype(L), 1)
        sign = B.sign(sign_z - 0.5)  # +1 or -1 with probability 0.5
        if ((dim % 2) == 1) or acceptance:
            samples.append(sign * proposal)
    samples = B.reshape(B.concat(*samples), *size)
    return key, B.cast(B.dtype(L), samples)


def spd_density_sample(key, size, params, degree, rho):
    nu = params["nu"]
    L = params["lengthscale"]

    samples = []
    while len(samples) < reduce(operator.mul, size, 1):
        key, X = B.randn(key, B.dtype(L), degree, degree)
        M = (X + B.transpose(X)) / 2

        eigv = eigvalsh(M)  # [D]

        # Note: 1.0 in safe_nu can be replaced by any finite positive value
        safe_nu = B.where(nu == np.inf, B.cast(B.dtype(L), np.r_[1.0]), nu)

        # for nu == np.inf
        proposal_nu_infinite = eigv / L

        # for nu < np.inf
        eigv_nu_finite = eigv * B.sqrt(2 * safe_nu / L**2 + B.sum(rho**2))
        # Gamma(nu, 2) distribution is the same as chi2(2nu) distribution
        key, chi2_sample = B.randgamma(key, B.dtype(L), 1, alpha=safe_nu, scale=2)
        chi_sample = B.sqrt(chi2_sample)
        proposal_nu_finite = eigv_nu_finite / chi_sample  # [D]

        proposal = B.where(nu == np.inf, proposal_nu_infinite, proposal_nu_finite)

        diffp = ordered_pairwise_differences(proposal)
        diffp = B.pi * B.abs(diffp)
        logprod = B.sum(B.log(B.tanh(diffp)), axis=-1)
        prod = B.exp(0.5 * logprod)
        assert B.all(prod > 0)

        # accept with probability `prod`
        key, u = B.rand(key, B.dtype(L), 1)
        acceptance = B.all(u < prod)
        if acceptance:
            samples.append(proposal)

    samples = B.reshape(B.concat(*samples), *size, degree)
    return key, samples
