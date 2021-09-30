"""
Hyperbolic space.
"""

import lab as B

from typing import Optional

from geometric_kernels.spaces import Space


def lorentz_distance(x1, x2, diag=False):
    """
    This function computes the Riemannian distance between points on a hyperbolic manifold.
    Parameters
    ----------
    :param x1: points on the hyperbolic manifold                                N1 x dim or b1 x ... x bk x N1 x dim
    :param x2: points on the hyperbolic manifold                                N2 x dim or b1 x ... x bk x N2 x dim
    Optional parameters
    -------------------
    :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`.
    Returns
    -------
    :return: matrix of manifold distance between the points in x1 and x2         N1 x N2 or b1 x ... x bk x N1 x N2
    """
    if diag is False:
        # Expand dimensions to compute all vector-vector distances
        x1 = B.expand_dims(x1, axis=-2)  # (..., N1, dim) -> (...,  N1, 1 dim)
        x2 = B.expand_dims(x2, axis=-3)  # (..., N2, dim) -> (...., 1, N2, dim)

        # Repeat x and y data along -2 and -3 dimensions to have b1 x ... x ndata_x x ndata_y x dim arrays
        # x1 = torch.cat(x2.shape[-2] * [x1], dim=-2)
        # x2 = torch.cat(x1.shape[-3] * [x2], dim=-3)

        # Difference between x1 and x2
        # diff_x = x1.view(-1, x1.shape[-1]) - x2.view(-1, x2.shape[-1])
        diff_X = x1 - x2  # (...,N1,N2, dim)

        # Compute the hyperbolic distance
        # mink_inner_prod = inner_minkowski_columns(diff_x.transpose(-1, -2), diff_x.transpose(-1, -2))
        mink_inner_prod = inner_minkowski_columns(diff_x, diff_x)  # (...,N1,N2, )
        mink_sqnorms = B.maximum(torch.zeros(mink_inner_prod), mink_inner_prod)  # (...,N1,N2, )
        mink_norms = B.sqrt(mink_sqnorms + 1e-8)
        # distance = 2 * B.arcsinh(.5 * mink_norms).view(x1.shape[:-1])
        distance = 2. * arcsinh(.5 * mink_norms)  # (..., N1, N2)

    else:
        assert x1 == x2
        # Difference between x1 and x2
        diff_x = x1 - x2  # (..., N, dim)

        # Compute the hyperbolic distance
        mink_inner_prod = inner_minkowski_columns(diff_x, diff_x)  # (..., N)
        mink_sqnorms = B.maximum(B.zeros(mink_inner_prod), mink_inner_prod)
        mink_norms = torch.sqrt(mink_sqnorms + 1e-8)
        distance = 2. * arcsinh(.5 * mink_norms)  # (..., N)

    return distance


def arcsinh(x):
    return B.log(x + B.sqrt(x**2 + 1))


def inner_minkowski_columns(x, y):
    # -x_0 * y_0 + x_1 * x_1 + ... + x_n * x_n
    return -x[..., 0] * y[..., 0] + B.sum(x[...,1:] * y[...,1:], axis=-1)  # (...,dim) -> (...)
    # return -x[0]*y[0] + B.sum(x[1:]*y[1:], axis=0)


def from_poincare_to_lorentz(x):
    raise NotImplementedError
    # x_torch = torch.tensor(x)
    first_coord = 1 + torch.pow(torch.linalg.norm(x_torch), 2)
    lorentz_point = torch.cat((first_coord.reshape(1), 2*x_torch)) / (1 - torch.pow(torch.linalg.norm(x_torch), 2))
    return lorentz_point


class Hyperbolic(Space):
    r"""
    Hyperbolic manifold.
    """
    def __init__(self, dim=1):
        self.dimension = dim

    def distance(self, x1: B.Numeric, x2: B.Numeric, diag: Optional[bool]=False) -> B.Numeric:
        assert (x1.shape[-1] == self.dimension) and (x2.shape[-1] == self.dimension)

        return lorentz_distance(x1, x2, diag=diag)

    def link_function(self, cosh_distance: B.Numeric, t: B.Numeric, b: B.Numeric, nu: B.Numeric, lengthscale: B.Numeric):
        """
        This function links the heat kernel to the Matérn kernel, i.e., the Matérn kernel correspond to the integral of
        this function from 0 to inf.
        Parameters
        ----------
        :param cosh_distance: precomputed cosine distance between the inputs
        :param t: heat kernel lengthscale
        :param b: heat kernel integral parameter
        Returns
        -------
        :return: link function between the heat and Matérn kernels
        """                      
        # Compute heat kernel integral part
        heat_kernel = B.exp((- b ** 2) / (2 * t)) * B.sinh(b) * B.sin(np.pi * b / t) / \
                      B.pow(B.cosh(b) + cosh_distance, (self.dimension + 1.) / 2.)

        result = B.pow(t, nu - 1.0) \
                 * B.exp(- 2.0 * nu / lengthscale ** 2 * t) \
                 * heat_kernel
        

        return result
