"""
Spaces for which there exist analytical expressions for the manifold
and the eigenvalues and functions. Examples include the `Circle` and the `Hypersphere`.
The Geomstats package is used for most of the geometric calculations.
"""

import geomstats as gs
import lab as B

from geometric_kernels.eigenfunctions import (
    Eigenfunctions,
    EigenfunctionWithAdditionTheorem,
)
from geometric_kernels.lab_extras import from_numpy
from geometric_kernels.spaces import DiscreteSpectrumSpace
from geometric_kernels.utils import Optional, chain

ATOL = 1e-12


class SinCosEigenfunctions(EigenfunctionWithAdditionTheorem):
    """
    Eigenfunctions Laplace-Beltrami operator on the circle correspond
    to the Fourier basis, i.e. sin and cosines..
    """

    def __init__(self, num_eigenfunctions: int) -> None:
        assert (
            num_eigenfunctions % 2 == 1
        ), "num_eigenfunctions needs to be odd to include all eigenfunctions within a level."
        assert num_eigenfunctions >= 1

        self._num_eigenfunctions = num_eigenfunctions
        # We know `num_eigenfunctions` is odd, therefore:
        self._num_levels = num_eigenfunctions // 2 + 1

    def __call__(self, X: B.Numeric, **parameters) -> B.Numeric:
        """
        :param X: polar coordinates on the circle, [N, 1].
        :param parameters: unused.
        """
        N = B.shape(X)[0]
        theta = X
        const = 2.0 ** 0.5
        values = []
        for level in range(self.num_levels):
            if level == 0:
                values.append(B.ones(X.dtype, N, 1))
            else:
                freq = 1.0 * level
                values.append(const * B.cos(freq * theta))
                values.append(const * B.sin(freq * theta))

        return B.concat(*values, axis=1)  # [N, M]

    def _addition_theorem(self, X: B.Numeric, X2: B.Numeric, **parameters) -> B.Numeric:
        r"""
        Returns the result of applying the additional theorem when
        summing over all the eigenfunctions within a level, for each level

        Concretely in the case for inputs on the sphere S^1:

        .. math:
            \sin(l \theta_1) \sin(l \theta_2) + \cos(l \theta_1) \cos(l \theta_2)
                = N_l \cos(l (\theta_1 - \theta_2)),
        where N_l = 1 for l = 0, else N_l = 2.

        :param X: [N, 1]
        :param X2: [N2, 1]
        :param parameters: unused.
        :return: Evaluate the sum of eigenfunctions on each level. Returns
            a value for each level [N, N2, L]
        """
        theta1, theta2 = X, X2
        angle_between = theta1[:, None, :] - theta2[None, :, :]  # [N, N2, 1]
        freqs = B.range(X.dtype, self.num_levels)  # [L]
        values = B.cos(freqs[None, None, :] * angle_between)  # [N, N2, L]
        values = (
            B.cast(
                X.dtype, from_numpy(X, self.num_eigenfunctions_per_level)[None, None, :]
            )
            * values
        )
        return values  # [N, N2, L]

    def _addition_theorem_diag(self, X: B.Numeric, **parameters) -> B.Numeric:
        """
        Returns the sum of eigenfunctions on a level for which we have a simplified expression

        :param X: [N, 1]
        :param parameters: unused.
        :return: Evaluate the sum of eigenfunctions on each level. Returns
            a value for each level [N, L]
        """
        N = X.shape[0]
        ones = B.ones(X.dtype, N, self.num_levels)  # [N, L]
        value = ones * B.cast(
            X.dtype, from_numpy(X, self.num_eigenfunctions_per_level)[None, :]
        )
        return value  # [N, L]

    @property
    def num_eigenfunctions(self) -> int:
        """Number of eigenfunctions, M"""
        return self._num_eigenfunctions

    @property
    def num_levels(self) -> int:
        """
        Number of levels, L

        For each level except the first where there is just one, there are two
        eigenfunctions.
        """
        return self._num_levels

    @property
    def num_eigenfunctions_per_level(self) -> B.Numeric:
        """Number of eigenfunctions per level, [N_l]_{l=0}^{L-1}"""
        return [1 if level == 0 else 2 for level in range(self.num_levels)]


class Circle(DiscreteSpectrumSpace, gs.geometry.hypersphere.Hypersphere):
    r"""
    Circle :math:`\mathbb{S}^1` manifold with sinusoids and cosines eigenfunctions.
    """

    def __init__(self):
        super().__init__(dim=1)

    def is_tangent(
        self,
        vector: B.Numeric,
        base_point: Optional[B.Numeric] = None,  # type: ignore
        atol: float = ATOL,
    ) -> bool:
        """
        Check whether the `vector` is tangent at `base_point`.
        :param vector: shape=[..., dim]
            Vector to evaluate.
        :param base_point: shape=[..., dim]
            Point on the manifold. Defaults to `None`.
        :param atol: float
            Absolute tolerance.
            Optional, default: 1e-6.
        :return: Boolean denoting if vector is a tangent vector at the base point.
        """
        raise NotImplementedError("`is_tangent` is not implemented for `Hypersphere`")

    @property
    def dimension(self) -> int:
        return 1

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        :param num: number of eigenfunctions returned.
        """
        return SinCosEigenfunctions(num)

    def get_eigenvalues(self, num: int) -> B.Numeric:
        """
        First `num` eigenvalues of the Laplace-Beltrami operator

        :return: [num, 1] array containing the eigenvalues
        """
        eigenfunctions = SinCosEigenfunctions(num)
        eigenvalues_per_level = B.range(eigenfunctions.num_levels) ** 2  # [L,]
        eigenvalues = chain(
            eigenvalues_per_level,
            eigenfunctions.num_eigenfunctions_per_level,
        )  # [num,]
        return B.reshape(eigenvalues, -1, 1)  # [num, 1]
