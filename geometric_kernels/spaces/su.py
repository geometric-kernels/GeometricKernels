"""
This module provides the :class:`SpecialUnitary` space and the representation
of its spectrum, the :class:`SUEigenfunctions` class.
"""

import itertools
import json
import math
import operator
from functools import reduce
from pathlib import Path

import lab as B
import numpy as np
from opt_einsum import contract as einsum

from geometric_kernels.lab_extras import (
    complex_conj,
    create_complex,
    dtype_double,
    from_numpy,
    qr,
)
from geometric_kernels.spaces.eigenfunctions import Eigenfunctions
from geometric_kernels.spaces.lie_groups import (
    LieGroupCharacter,
    MatrixLieGroup,
    WeylAdditionTheorem,
)
from geometric_kernels.utils.utils import chain


class SUEigenfunctions(WeylAdditionTheorem):
    def __init__(self, n, num_levels, compute_characters=True):
        self.n = n
        self.dim = n * (n - 1)
        self.rank = n - 1

        self.rho = np.arange(self.n - 1, -self.n, -2) * 0.5

        super().__init__(n, num_levels, compute_characters)

    def _generate_signatures(self, num_levels):
        sign_vals_lim = 100 if self.n in (1, 2) else 30 if self.n == 3 else 10
        signatures = list(
            itertools.combinations_with_replacement(
                range(sign_vals_lim, -1, -1), r=self.rank
            )
        )
        signatures = [sgn + (0,) for sgn in signatures]

        dimensions = [self._compute_eigenvalue(signature) for signature in signatures]
        min_ind = np.argpartition(dimensions, num_levels)[:num_levels]
        signatures = [signatures[i] for i in min_ind]
        return signatures

    def _compute_dimension(self, signature):
        rep_dim = reduce(
            operator.mul,
            (
                reduce(
                    operator.mul,
                    (
                        signature[i - 1] - signature[j - 1] + j - i
                        for j in range(i + 1, self.n + 1)
                    ),
                )
                / math.factorial(self.n - i)
                for i in range(1, self.n)
            ),
        )
        return int(round(rep_dim))

    def _compute_eigenvalue(self, signature):
        normalized_signature = signature - np.mean(signature)
        lb_eigenvalue = (
            np.linalg.norm(self.rho + normalized_signature) ** 2
            - np.linalg.norm(self.rho) ** 2
        )
        return lb_eigenvalue

    def _compute_character(self, n, signature):
        return SUCharacter(n, signature)

    def _torus_representative(self, X):
        return B.eig(X, False)

    def inverse(self, X: B.Numeric) -> B.Numeric:
        return B.transpose(X).conj()

    @property
    def num_levels(self) -> int:
        """Number of levels, L"""
        return self._num_levels

    @property
    def num_eigenfunctions_per_level(self) -> int:
        """Number of eigenfunctions per level"""
        return self._dimensions


class SUCharacter(LieGroupCharacter):
    def __init__(self, n, signature):
        self.signature = signature
        self.n = n
        self.coeffs, self.monoms = self._load()

    def _load(self):
        group_name = "SU({})".format(self.n)
        file_path = Path(__file__).with_name("precomputed_characters.json")
        with file_path.open("r") as file:
            character_formulas = json.load(file)
            try:
                cs, ms = character_formulas[group_name][str(self.signature)]
                coeffs, monoms = (np.array(data) for data in (cs, ms))
                return coeffs, monoms
            except KeyError as e:
                raise KeyError(
                    "Unable to retrieve character parameters for signature {} of {}, "
                    "perhaps it is not precomputed."
                    "Run compute_characters.py with changed parameters.".format(
                        e.args[0], group_name
                    )
                ) from None

    def __call__(self, gammas):
        char_val = B.zeros(B.dtype(gammas), *gammas.shape[:-1])
        for coeff, monom in zip(self.coeffs, self.monoms):
            char_val += coeff * B.prod(gammas ** from_numpy(gammas, monom), axis=-1)
        return char_val


class SpecialUnitary(MatrixLieGroup):
    r"""
    The GeometricKernels space representing the special unitary group
    :math:`SU(n)` consisting of n by n complex unitary matrices with unit
    determinant.

    The elements of this space are represented as :math:`n \times n` unitary
    matrices with complex entries and unit determinant.

    Note: we only support n >= 2. Mathematically, SU(1) is trivial, consisting
    of a single element (the identity), chances are you do not need it.
    For large values of n, you might need to run the `compute_characters.py`
    script to precompute the necessary mathematical quantities beyond the ones
    provided by default.
    """

    def __init__(self, n):
        if n < 2:
            raise ValueError(f"Only n >= 2 is supported. n = {n} was provided.")
        self.n = n
        self.dim = n * (n - 1) // 2
        super().__init__()

    @property
    def dimension(self) -> int:
        return self.dim

    def inverse(self, X: B.Numeric) -> B.Numeric:
        return complex_conj(B.transpose(X))

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        :param num: number of eigenfunctions returned.
        """
        return SUEigenfunctions(self.n, num)

    def get_eigenvalues(self, num: int) -> B.Numeric:
        """
        Eigenvalues of first 'num' levels of the Laplace-Beltrami operator.

        :return: [num, 1] array containing the eigenvalues
        """
        eigenfunctions = SUEigenfunctions(self.n, num)
        eigenvalues = np.array(
            [eigenvalue for eigenvalue in eigenfunctions._eigenvalues]
        )
        return B.reshape(eigenvalues, -1, 1)  # [num, 1]

    def get_repeated_eigenvalues(self, num: int) -> B.Numeric:
        """Eigenvalues of first 'num' levels of the Laplace-Beltrami operator,
        repeated according to their multiplicity.

        :return: [M, 1] array containing the eigenvalues
        """
        eigenfunctions = SUEigenfunctions(self.n, num)
        eigenvalues = chain(
            eigenfunctions._eigenvalues,
            [rep_dim**2 for rep_dim in eigenfunctions._dimensions],
        )
        return B.reshape(eigenvalues, -1, 1)  # [M, 1]

    def random(self, key, number):
        if self.n == 2:
            # explicit parametrization via the double cover SU(2) = S^3
            key, sphere_point = B.random.randn(key, dtype_double(key), number, 4)
            sphere_point /= B.reshape(
                B.sqrt(einsum("ij,ij->i", sphere_point, sphere_point)), -1, 1
            )
            a = create_complex(sphere_point[..., 0], sphere_point[..., 1])
            b = create_complex(sphere_point[..., 2], sphere_point[..., 3])
            r1 = B.stack(a, -complex_conj(b), axis=-1)
            r2 = B.stack(b, complex_conj(a), axis=-1)
            q = B.stack(r1, r2, axis=-1)
            return key, q
        else:
            key, real = B.random.randn(key, dtype_double(key), number, self.n, self.n)
            key, imag = B.random.randn(key, dtype_double(key), number, self.n, self.n)
            h = create_complex(real, imag) / B.sqrt(2)
            q, r = qr(h, mode="complete")
            r_diag = einsum("...ii->...i", r)
            r_diag_inv_phase = complex_conj(r_diag / B.abs(r_diag))
            q *= r_diag_inv_phase[:, None]
            q_det = B.det(q)
            q_det_inv_phase = complex_conj((q_det / B.abs(q_det)))
            q[:, :, 0] *= q_det_inv_phase[:, None]
            return key, q
