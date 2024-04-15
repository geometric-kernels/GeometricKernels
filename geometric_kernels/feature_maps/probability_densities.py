"""
This module provide the routines for sampling from the Gaussian and Student-t
probability densities in a backend-agnostic way. It also provides the routines
for sampling the non-standard probability densities that arise in relation to
the :class:`~.spaces.Hyperbolic` and
:class:`~.spaces.SymmetricPositiveDefiniteMatrices` spaces.
"""

import operator
from functools import reduce

import lab as B
import numpy as np
from beartype.typing import Dict, List, Optional, Tuple
from sympy import Poly, Product, symbols

from geometric_kernels.lab_extras import (
    cumsum,
    dtype_double,
    dtype_integer,
    eigvalsh,
    from_numpy,
)
from geometric_kernels.utils.utils import ordered_pairwise_differences


def student_t_sample(
    key: B.RandomState,
    size: B.Numeric,
    deg_freedom: B.Numeric,
    dtype: Optional[B.DType] = None,
) -> Tuple[B.RandomState, B.Numeric]:
    r"""
    Sample from the Student-t distribution with `deg_freedom` degrees of
    freedom, using `key` random state, returning sample of shape `size`.

    Student-t random variable with `nu` degrees of freedom can be represented
    as $T=\frac{Z}{\sqrt{V/\nu}}$, where `Z` is the standard Gaussian random
    variable and `V` is a $\chi^2(\nu)$ random variable. The $\chi^2(\nu)$
    distribution is the same as $\Gamma(\nu / 2, 2)$ distribution, and
    therefore $V/\nu \sim \Gamma(\nu / 2, 2 / \nu)$. We use these properties
    to sample the Student-t random variable.

    :param key: either `np.random.RandomState`, `tf.random.Generator`,
        `torch.Generator` or `jax.tensor` (representing random state).
    :param size: shape of the returned sample.
    :param deg_freedom: the number of degrees of freedom of the Student-t
        distribution.
    :param dtype: dtype of the returned tensor.
    :return: `Tuple(key, samples)` where `samples` is a `size`-shaped array of
        samples of type `dtype`, and `key` is the updated random key for `jax`,
        or the similar random state (generator) for any other backend.
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


def base_density_sample(
    key: B.RandomState,
    size: B.Numeric,
    params: Dict[str, B.Numeric],
    dim: int,
    rho: B.Numeric,
) -> Tuple[B.RandomState, B.Numeric]:
    r"""
    The Matérn kernel's spectral density is $p_{\nu,\kappa}(\lambda)$, where
    $\nu$ is the smoothness parameter, $\kappa$ is the length scale and
    $p_{\nu,\kappa}$ is the Student-t or Gaussian density, depending on the
    smoothness.

    We call it "base density" and this function returns a sample from it.

    :param key: either `np.random.RandomState`, `tf.random.Generator`,
        `torch.Generator` or `jax.tensor` (representing random state).
    :param size: shape of the returned sample.
    :param params: params of the kernel.
    :param dim: dimensionality of the space the kernel is defined on.
    :param rho: `rho` vector of the space.
    :return: `Tuple(key, samples)` where `samples` is a `size`-shaped array of
        samples, and `key` is the updated random key for `jax`, or the similar
        random state (generator) for any other backend.
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
    # sample from the Student-t with 2\nu + dim(space) - dim(rho)  degrees of freedom
    deg_freedom = 2 * safe_nu + dim - B.rank(rho)
    key, u_nu_finite = student_t_sample(key, size, deg_freedom, B.dtype(L))

    u = B.where(nu == np.inf, u_nu_infinite, u_nu_finite)

    scale_nu_infinite = L
    scale_nu_finite = L / B.sqrt(
        safe_nu / deg_freedom + B.sum(rho**2) * L**2 / (2 * deg_freedom)
    )

    scale = B.where(nu == np.inf, scale_nu_infinite, scale_nu_finite)

    scale = B.cast(B.dtype(u), scale)
    return key, u / scale


def _randcat_fix(
    key: B.RandomState, dtype: B.DType, size: B.Numeric, p: B.Numeric
) -> Tuple[B.RandomState, B.Numeric]:
    """
    Sample from the categorical variable with probabilities `p`.

    :param key: either `np.random.RandomState`, `tf.random.Generator`,
        `torch.Generator` or `jax.tensor` (representing random state).
    :param dtype: dtype of the returned tensor.
    :param size: shape of the returned sample.
    :param p: vector of (potentially unnormalized) probabilities
        defining the discrete distribution to sample from.
    :return: `Tuple(key, samples)` where `samples` is a `size`-shaped array of
        samples of type `dtype`, and `key` is the updated random key for `jax`,
        or the similar random state (generator) for any other backend.
    """
    p = p / B.sum(p, axis=-1, squeeze=False)
    # Perform sampling routine.
    cdf = cumsum(p, axis=-1)
    key, u = B.rand(key, dtype, size, *B.shape(p)[:-1])
    inds = B.argmax(B.cast(dtype_integer(key), u[..., None] < cdf[None]), axis=-1)
    return key, B.cast(dtype, inds)


def _alphas(n: int) -> B.Numeric:
    r"""
    Compute alphas for Prop. 16 & 17 of cite:t:`azangulov2023`
    for the hyperbolic space of dimension `n`.

    :param n: dimension of the hyperbolic space, n >= 2.
    :return: array of alphas.

    .. todo:: precompute these, rather than computing in runtime.

    .. todo:: update proposition numbers when the paper gets published.
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


def _sample_mixture_heat(
    key: B.RandomState, alpha: B.Numeric, lengthscale: B.Numeric
) -> Tuple[B.RandomState, B.Numeric]:
    r"""
    Sample from the mixture distribution from Prop. 16 for specific alphas
    `alpha` and length scale (kappa) `lengthscale` using `key` random state.

    :param key: either `np.random.RandomState`, `tf.random.Generator`,
                `torch.Generator` or `jax.tensor` (representing random state).
    :param alpha: unnormalized coefficients of the mixture.
    :param lengthscale: length scale (kappa).
    :return: `Tuple(key, sample)` where `sample` is a single sample from the
        distribution, and `key` is the updated random key for `jax`, or the
        similar random state (generator) for any other backend.

    .. todo::
       do we need reparameterization trick for hyperparameter optimization?

    .. todo:: update proposition numbers when the paper gets published.
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
    key, ind = _randcat_fix(key, dtype, 1, cs)

    # Gamma(nu/2, 2) distribution is the same as chi2(nu) distribution
    key, s = B.randgamma(key, dtype, 1, alpha=(ind + 1) / 2, scale=2)
    s = B.sqrt(s) / lengthscale
    return key, s


def _sample_mixture_matern(
    key: B.RandomState,
    alpha: B.Numeric,
    lengthscale: B.Numeric,
    nu: B.Numeric,
    dim: int,
) -> Tuple[B.RandomState, B.Numeric]:
    r"""
    Sample from the mixture distribution from Prop. 17 of
    cite:t:`azangulov2023` for specific alphas `alpha`, length
    scale (kappa) `lengthscale`, smoothness `nu` and dimension `dim`, using
    `key` random state.

    :param key: either `np.random.RandomState`, `tf.random.Generator`,
                `torch.Generator` or `jax.tensor` (representing random state).
    :param alpha: unnormalized coefficients of the mixture.
    :param lengthscale: length scale (kappa).
    :param nu: smoothness parameter of Matérn kernels.
    :param dim: dimension of the hyperbolic space.
    :return: `Tuple(key, sample)` where `sample` is a single sample from the
        distribution, and `key` is the updated random key for `jax`, or the
        similar random state (generator) for any other backend.

    .. todo::
       do we need reparameterization trick for hyperparameter optimization?

    .. todo:: update proposition numbers when the paper gets published.
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

    key, ind = _randcat_fix(key, dtype, 1, cs)
    key, p = B.randbeta(
        key, dtype, 1, alpha=(ind + 1) / 2, beta=nu + (dim - ind - 1) / 2
    )
    p = p / (1 - p)  # beta prime distribution
    s = B.sqrt(gamma * p)
    return key, s


def hyperbolic_density_sample(
    key: B.RandomState, size: B.Numeric, params: Dict[str, B.Numeric], dim: int
) -> Tuple[B.RandomState, B.Numeric]:
    r"""
    This function samples from the full (i.e., including the $c(\lambda)^{-2}$
    factor, where $c$ is the Harish-Chandra's $c$-function) spectral density of
    the heat/Matérn kernel on the hyperbolic space, using rejection sampling.

    :param key: either `np.random.RandomState`, `tf.random.Generator`,
                `torch.Generator` or `jax.tensor` (representing random state).
    :param size: shape of the returned sample.
    :param params: params of the kernel.
    :param dim: dimensionality of the space the kernel is defined on.
    :return: `Tuple(key, samples)` where `samples` is a `size`-shaped array of
        samples, and `key` is the updated random key for `jax`, or the similar
        random state (generator) for any other backend.
    """
    assert "nu" in params
    assert "lengthscale" in params

    nu = params["nu"]
    L = params["lengthscale"]

    alpha = _alphas(dim)

    def base_sampler(key):
        # Note: 1.0 in safe_nu can be replaced by any finite positive value
        safe_nu = B.where(nu == np.inf, B.cast(B.dtype(L), np.r_[1.0]), nu)

        # for nu == np.inf
        key, sample_mixture_nu_infinite = _sample_mixture_heat(key, alpha, L)
        # for nu < np.inf
        key, sample_mixture_nu_finite = _sample_mixture_matern(
            key, alpha, L, safe_nu, dim
        )

        return key, B.where(
            nu == np.inf, sample_mixture_nu_infinite, sample_mixture_nu_finite
        )

    samples: List[B.Numeric] = []
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


def spd_density_sample(
    key: B.RandomState,
    size: B.Numeric,
    params: Dict[str, B.Numeric],
    degree: int,
    rho: B.Numeric,
) -> Tuple[B.RandomState, B.Numeric]:
    r"""
    This function samples from the full (i.e., including the $c(\lambda)^{-2}$
    factor, where $c$ is the Harish-Chandra's $c$-function) spectral density
    of the heat/Matérn kernel on the manifold of symmetric positive definite
    matrices, using rejection sampling.

    :param key: either `np.random.RandomState`, `tf.random.Generator`,
                `torch.Generator` or `jax.tensor` (representing random state).
    :param size: shape of the returned sample.
    :param params: params of the kernel.
    :param dim: dimensionality of the space the kernel is defined on.
    :return: `Tuple(key, samples)` where `samples` is a `size`-shaped array of
        samples, and `key` is the updated random key for `jax`, or the similar
        random state (generator) for any other backend.
    """
    nu = params["nu"]
    L = params["lengthscale"]

    samples: List[B.Numeric] = []
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
