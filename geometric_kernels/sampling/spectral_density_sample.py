"""
Sampling from the spectral measure.
"""
import lab as B
import numpy as np

from geometric_kernels.lab_extras import dtype_double


def spectral_density_sample(key, size, params, dim):
    assert "nu" in params
    assert "lengthscale" in params

    nu = params["nu"]
    L = params["lengthscale"]

    if nu == np.inf:
        # sample from Gaussian
        key, u = B.randn(key, dtype_double(key), *size)
        return key, u / L
    elif nu > 0:
        # sample from the student-t with 2\nu + dim - 1 degrees of freedom
        deg_freedom = 2 * nu + dim - 1
        key, z = B.randn(key, dtype_double(key), *size)  # [O, 1]
        key, g = B.randgamma(
            key,
            dtype_double(key),
            *size,
            alpha=deg_freedom / 2,
            scale=2 / deg_freedom,
        )

        u = z / B.sqrt(g)
        return key, u / L
