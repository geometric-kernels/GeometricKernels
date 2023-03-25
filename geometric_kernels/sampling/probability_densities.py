"""
Sampling from the Gaussian and Student-t probability densities,
backend-agnostic.
"""
import lab as B
import numpy as np

from geometric_kernels.lab_extras import dtype_double


def student_t_sample(key, size, deg_freedom, dtype=None):
    r"""
    Sample from the Student-t distribution with `deg_freedom` degrees of freedom,
    using `key` random state, returning sample of the shape `size`.

    Studen-t random variable with `nu` degrees of freedom can be represented as
    :math:`T=\frac{Z}{\sqrt{V/\nu}}`, where `Z` is the standard normal r.v. and
    `V` is :math:`\chi^2(\nu)` r.v. The :math:`\chi^2(\nu)` distribution is the
    same as `\Gamma(\nu / 2, 2)` distribution, and therefore
    :math:`V/\nu \sim \Gamma(\nu / 2, 2 * \nu)`

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
    return u


def base_density_sample(key, size, params, dim):
    r"""
    The Matern kernel's spectral density is of the form
    :math:`c(\lambda) p_{\nu,\kappa}(\lambda)`,
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
        u = student_t_sample(key, size, deg_freedom)

    return key, u / L
