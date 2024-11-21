"""
Implements alternative formulas for the heat kernel on the hyperbolic manifold.

More specifically, the function :func:`hyperbolic_heat_kernel_odd` implements
analytic formulas for the heat kernel on odd-dimensional hyperbolic spaces. The
function :func:`hyperbolic_heat_kernel_even` implements a semi-analytic formula
for the heat kernel on even-dimensional hyperbolic spaces.

The implementation is provided mainly for testing purposes. Hypothetically, the
odd-dimensional formula could be used in practice, but the even-dimensional one
is not recommended due to its inefficiency and numerical instability.
"""

import lab as B
import numpy as np
import scipy
from beartype.typing import Optional

from geometric_kernels.lab_extras import cosh, sinh
from geometric_kernels.utils.manifold_utils import hyperbolic_distance


def hyperbolic_heat_kernel_odd(
    dim: int,
    t: float,
    X: B.Numeric,
    X2: Optional[B.Numeric] = None,
) -> B.Numeric:
    """
    The analytic formula for the heat kernel on the hyperbolic space of odd
    dimension, normalized to have k(x, x) = 1 for all x.

    :param t:
        The time parameter, a positive float.
    :param X:
        A batch of inputs, an array of shape [N, dim+1].
    :param X2:
        A batch of inputs, an array of shape [N2, dim+1]. If None, defaults to X.

    :return:
        The kernel matrix, an array of shape [N, N2]. The kernel is normalized,
        i.e. k(x, x) = 1 for all x.
    """

    if dim % 2 == 0:
        raise ValueError(
            "This function is only defined for odd-dimensional hyperbolic spaces. For even-dimensional spaces, use `hyperbolic_heat_kernel_even`."
        )

    if X2 is None:
        X2 = X

    dists = hyperbolic_distance(X, X2)
    dists_are_small = dists < 0.1

    if dim == 3:
        # This formula is taken from :cite:t:`azangulov2024b`, Equation (42).
        analytic_expr = dists / sinh(dists) * B.exp(-(dists**2) / (4 * t))

        # To get the Taylor expansion below, which gives a stable way to compute
        # the kernel for small distances, use the following Mathematica code:
        # > Subscript[F, 3][r_, t_] := r/Sinh[r]*Exp[-r^2/(4*t)]
        # > Series[Subscript[F, 3][r, t],{r, 0, 5}]
        taylor = 1 - (1 / 6 + 1 / (4 * t)) * dists**2

        return B.where(dists_are_small, taylor, analytic_expr)
    elif dim == 5:
        # The following expression follows from the recursive formula in Equation
        # (43) of :cite:t:`azangulov2024b`. In order to get the form below, you
        # can continue the Mathematica code above with the following:
        # > Subscript[U, 5][r_,t_] := -1/Sinh[r]*(D[Subscript[F, 3][rr,t ], rr] /. rr -> r)
        # > Subscript[C, 5][t_] := Limit[Subscript[U, 5][r, t], r -> 0]
        # > Subscript[F, 5][r_, t_] := Subscript[U, 5][r, t]/Subscript[C, 5][t]
        # > Simplify[Subscript[F, 5][r, t]]
        a = 3 * B.exp(-(dists**2) / (4 * t)) / (3 + 2 * t)
        b = dists**2 - 2 * t + 2 * dists * t * cosh(dists) / sinh(dists)
        c = 1.0 / sinh(dists) ** 2
        analytic_expr = a * b * c

        # To get the Taylor expansion below, which gives a stable way to compute
        # the kernel for small distances, use the following Mathematica code:
        # > Series[Subscript[F, 5][r, t], {r, 0, 5}]
        taylor = 1 - (15 - 30 * t - 16 * t**2) / (20 * t * (3 + 2 * t)) * dists**2

        return B.where(dists_are_small, taylor, analytic_expr)
    elif dim == 7:
        # The following expression follows from the recursive formula in Equation
        # (43) of :cite:t:`azangulov2024b`. In order to get the form below, you
        # can continue the Mathematica code above with the following:
        # > Subscript[U, 7][r_, t_] := -1/Sinh[r]*(D[Subscript[F, 5][rr, t ], rr] /. rr -> r)
        # > Subscript[C, 7][t_]:=Limit[Subscript[U, 7][r, t],r->0]
        # > Subscript[F, 7][r_, t_] := Subscript[U, 7][r, t]/Subscript[C, 7][t]
        # > TrigFactor[Subscript[F, 7][r,t]]
        a = 15 * B.exp(-(dists**2) / (4 * t)) / (2 * (15 + 30 * t + 16 * t**2))
        b1 = -12 * t**2 * sinh(2 * dists)
        b2 = (6 * t + 16 * t**2 + (8 * t**2 - 6 * t) * cosh(2 * dists)) * dists
        b3 = 6 * t * sinh(2 * dists) * dists**2
        b4 = (cosh(2 * dists) - 1) * dists**3
        b = b1 + b2 + b3 + b4
        analytic_expr = a * b / sinh(dists) ** 5

        # To get the Taylor expansion below, which gives a stable way to compute
        # the kernel for small distances, use the following Mathematica code:
        # > Series[Subscript[F, 7][r, t],{r, 0, 5}]
        taylor = (
            1
            - 3
            * (35 + 140 * t + 196 * t**2 + 96 * t**3)
            / (28 * t * (15 + 30 * t + 16 * t**2))
            * dists**2
        )

        return B.where(dists_are_small, taylor, analytic_expr)
    else:
        raise NotImplementedError(
            f"Odd-dimensional hyperbolic space of dimension {dim} is not supported."
        )


def _hyperbolic_heat_kernel_2d_unnormalized(t: float, rho: float) -> float:
    def integrand(t: float, s: float, rho: float) -> float:
        result = (s + rho) * np.exp(-((s + rho) ** 2) / (4 * t))
        result /= np.sqrt((np.cosh(s + rho) - np.cosh(rho)))
        return result

    integral, error = scipy.integrate.quad(lambda s: integrand(t, s, rho), 0, np.inf)

    return integral


def hyperbolic_heat_kernel_even(
    dim: int,
    t: float,
    X: B.NPNumeric,
    X2: Optional[B.NPNumeric] = None,
) -> B.NPNumeric:

    if dim % 2 != 0:
        raise ValueError(
            "This function is only defined for even-dimensional hyperbolic spaces. For odd-dimensional spaces, use `hyperbolic_heat_kernel_odd`."
        )
    elif dim != 2:
        # The integrand in higher dimensions may be obtained from the
        # recursive formula in Equation (43) of :cite:t:`azangulov2024b`.
        #
        # For example, to get the integrand in dimension 4, you can use the
        # following Mathematica code:
        # > Subscript[F, 2][r_,t_,s_]:=((r+s)*Exp[-(r+s)^2/(4*t)])/(Cosh[r+s]-Cosh[r])^(1/2)
        # > Subscript[F, 4][r_,t_,s_]:=-(1/Sinh[r])*(D[Subscript[F, 2][rr,t,s ], rr] /. rr -> r)
        # > Simplify[Subscript[F, 4][r, t, s]]
        #
        # However, in higher dimensions, these integrands become a huge pain
        # to work with. For example, the integrand in dimension 4 looks like
        #
        #  |       ***
        #  |     *       **
        #  |    *           ***
        #  |   *                ***
        #  |   *                     ****
        # -|- * -------------------------
        #  |  *
        #  |  *
        #  | *
        #
        # This function is very hard to numerically integrate because
        # * it has a singularity at s=0 (it explodes to -inf),
        # * it changes sign,
        # * the domain of integration is infinite, and for large s you get 0*inf
        #   situations all the time (=> you have to use an asymptotic for s->inf),
        # * finally, it also behaves very poorly for small rho (=> you have to
        #   use another asymptotic for rho->0).
        #
        # This is why we don't provide the integrand in higher dimensions.
        #
        # To whoever wants to implement it in higher dimensions in the future:
        # Don't. It's not worth it. The Fourier feature approximation is already
        # very accurate and, crucially, it is numerically stable. Furthermore,
        # any implementation of the integrand in higher dimensions will be very
        # hacky, thus diminishing its value for testing purposes.
        raise NotImplementedError(
            f"Even-dimensional hyperbolic space of dimension {dim} is not supported. See the comments in the code for more details on why."
        )

    if X2 is None:
        X2 = X

    normalization = _hyperbolic_heat_kernel_2d_unnormalized(t, 0)

    result = np.zeros((X.shape[0], X2.shape[0]))

    # This is a very inefficient implementation, but it will do for tests.
    for i, x in enumerate(X):
        for j, x2 in enumerate(X2):
            rho = hyperbolic_distance(x, x2).squeeze()
            cur_result = _hyperbolic_heat_kernel_2d_unnormalized(t, rho)
            result[i, j] = cur_result / normalization

    return result
