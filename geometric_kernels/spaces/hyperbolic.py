"""
Hyperbolic space.
"""

from typing import Optional

import geomstats as gs
import lab as B

from geometric_kernels.lab_extras import cosh, trapz
from geometric_kernels.spaces import Space


class Hyperbolic(Space, gs.geometry.hyperboloid.Hyperboloid):
    r"""
    Hyperbolic manifold.
    """

    def __init__(self, dim=1):
        super().__init__(dim=dim)

    @property
    def dimension(self) -> int:
        return self.dim

    @property
    def is_compact(self) -> bool:
        return False

    def distance(
        self, x1: B.Numeric, x2: B.Numeric, diag: Optional[bool] = False
    ) -> B.Numeric:
        assert B.all(self.belongs(x1)) and B.all(self.belongs(x2))

        return self.metric.dist(x1, x2)

    def heat_kernel(
        self, distance: B.Numeric, t: B.Numeric, num_points: int = 100
    ) -> B.Numeric:
        """
        Compute the heat kernel associated with the space.

        We use Millson's formula for the heat kernel.

        References:
        [1] A. Grigoryan and M. Noguchi,
        The heat kernel on hyperbolic space.
        Bulletin of the London Mathematical Society, 30(6):643–650, 1998.

        Parameters
        ----------
        :param distance: precomputed distance between the inputs
        :param t: heat kernel lengthscale
        :param num_points: number of points in the integral
        Returns
        -------
        :return: heat kernel values
        """
        if self.dimension == 1:
            heat_kernel = B.exp(
                -B.power(distance[..., None], 2) / (4 * t)
            )  # (..., N1, N2, T)

        elif self.dimension == 2:
            expanded_distance = B.expand_dims(
                B.expand_dims(distance, -1), -1
            )  # (... N1, N2) -> (..., N1, N2, 1, 1)

            # TODO: the behavior of this kernel is not so stable around zero distance
            # due to the division in the computation of the integral value and
            # depends on the start of the s_vals interval
            s_vals = (
                B.linspace(1.5e-1, 100.0, num_points) + expanded_distance
            )  # (..., N1, N2, 1, S)
            integral_vals = (
                s_vals
                * B.exp(-(s_vals ** 2) / (4 * t[:, None]))
                / B.sqrt(cosh(s_vals) - cosh(expanded_distance))
            )  # (..., N1, N2, T, S)

            heat_kernel = trapz(integral_vals, s_vals, axis=-1)  # (..., N1, N2, T)

        elif self.dimension == 3:
            heat_kernel = B.exp(
                -B.power(distance[..., None], 2) / (4 * t)
            )  # (..., N1, N2, T)
            heat_kernel = (
                heat_kernel * (distance + 1e-8) / B.sinh(distance + 1e-8)
            )  # Adding 1e-8 avoids numerical issues around d=0

        else:
            raise NotImplementedError

        return heat_kernel
