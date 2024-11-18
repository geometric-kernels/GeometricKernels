"""
This module provides the :class:`SpecialUnitary` space and the respective
:class:`~.eigenfunctions.Eigenfunctions` subclass :class:`SUEigenfunctions`.
"""

import itertools
import json
import math
import operator
from functools import reduce

import lab as B
import numpy as np
from beartype.typing import List, Tuple

from geometric_kernels.lab_extras import (
    complex_conj,
    complex_like,
    create_complex,
    dtype_double,
    from_numpy,
    qr,
)
from geometric_kernels.spaces.eigenfunctions import Eigenfunctions
from geometric_kernels.spaces.lie_groups import (
    CompactMatrixLieGroup,
    LieGroupCharacter,
    WeylAdditionTheorem,
)
from geometric_kernels.utils.utils import chain, get_resource_file_path


class SUEigenfunctions(WeylAdditionTheorem):
    def __init__(self, n, num_levels, compute_characters=True):
        self.n = n
        self.dim = n**2 - 1
        self.rank = n - 1

        self.rho = np.arange(self.n - 1, -self.n, -2) * 0.5

        super().__init__(n, num_levels, compute_characters)

    def _generate_signatures(self, num_levels: int) -> List[Tuple[int, ...]]:
        sign_vals_lim = 100 if self.n in (1, 2) else 30 if self.n == 3 else 10
        signatures = list(
            itertools.combinations_with_replacement(
                range(sign_vals_lim, -1, -1), r=self.rank
            )
        )
        signatures = [sgn + (0,) for sgn in signatures]

        eig_and_signature = [
            (
                round(4 * (len(signature) ** 2) * self._compute_eigenvalue(signature)),
                signature,
            )
            for signature in signatures
        ]

        eig_and_signature.sort()
        signatures = [eig_sgn[1] for eig_sgn in eig_and_signature][:num_levels]
        return signatures

    def _compute_dimension(self, signature: Tuple[int, ...]) -> int:
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

    def _compute_eigenvalue(self, signature: Tuple[int, ...]) -> B.Float:
        normalized_signature = np.asarray(signature, dtype=np.float64) - np.mean(
            signature
        )
        lb_eigenvalue = (
            np.linalg.norm(self.rho + normalized_signature) ** 2
            - np.linalg.norm(self.rho) ** 2
        )
        return lb_eigenvalue

    def _compute_character(
        self, n: int, signature: Tuple[int, ...]
    ) -> LieGroupCharacter:
        return SUCharacter(n, signature)

    def _torus_representative(self, X):
        return B.eig(X, False)

    def inverse(self, X: B.Numeric) -> B.Numeric:
        return SpecialUnitary.inverse(X)


class SUCharacter(LieGroupCharacter):
    """
    The class that represents a character of the SU(n) group.

    Many of the characters on SU(n) are complex-valued.

    These are polynomials whose coefficients are precomputed and stored in a
    file. By default, there are 20 precomputed characters for n from 2 to 6.
    If you want more, use the `compute_characters.py` script.

    :param n:
        The order n of the SO(n) group.
    :param signature:
        The signature that determines a particular character (and an
        irreducible unitary representation along with it).
    """

    def __init__(self, n: int, signature: Tuple[int, ...]):
        self.signature = signature
        self.n = n
        self.coeffs, self.monoms = self._load()

    def _load(self):
        group_name = "SU({})".format(self.n)
        with get_resource_file_path("precomputed_characters.json") as file_path:
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

    def __call__(self, gammas: B.Numeric) -> B.Numeric:
        char_val = B.zeros(B.dtype(gammas), *gammas.shape[:-1])
        for coeff, monom in zip(self.coeffs, self.monoms):
            char_val += coeff * B.prod(
                B.power(
                    gammas, B.cast(complex_like(gammas), from_numpy(gammas, monom))
                ),
                axis=-1,
            )
        return char_val


class SpecialUnitary(CompactMatrixLieGroup):
    r"""
    The GeometricKernels space representing the special unitary group SU(n)
    consisting of n by n complex unitary matrices with unit determinant.

    The elements of this space are represented as n x n unitary
    matrices with complex entries and unit determinant.

    .. note::
        A tutorial on how to use this space is available in the
        :doc:`SpecialUnitary.ipynb </examples/SpecialUnitary>` notebook.

    :param n:
        The order n of the group SU(n).

    .. note::
        We only support n >= 2. Mathematically, SU(1) is trivial, consisting
        of a single element (the identity), chances are you do not need it.
        For large values of n, you might need to run the `compute_characters.py`
        script to precompute the necessary mathematical quantities beyond the
        ones provided by default.

    .. admonition:: Citation

        If you use this GeometricKernels space in your research, please consider
        citing :cite:t:`azangulov2024a`.
    """

    def __init__(self, n: int):
        if n < 2:
            raise ValueError(f"Only n >= 2 is supported. n = {n} was provided.")
        self.n = n
        self.dim = n**2 - 1
        self.rank = n - 1
        super().__init__()

    def __str__(self):
        return f"SpecialUnitary({self.n})"

    @property
    def dimension(self) -> int:
        """
        The dimension of the space, as that of a Riemannian manifold.

        :return:
            floor(n^2-1) where n is the order of the group SU(n).
        """
        return self.dim

    @staticmethod
    def inverse(X: B.Numeric) -> B.Numeric:
        return complex_conj(
            B.transpose(X)
        )  # B.transpose only inverses the last two dims.

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        Returns the :class:`~.SUEigenfunctions` object with `num` levels
        and order n.

        :param num:
            Number of levels.
        """
        return SUEigenfunctions(self.n, num)

    def get_eigenvalues(self, num: int) -> B.Numeric:
        eigenfunctions = SUEigenfunctions(self.n, num)
        eigenvalues = np.array(
            [eigenvalue for eigenvalue in eigenfunctions._eigenvalues]
        )
        return B.reshape(eigenvalues, -1, 1)  # [num, 1]

    def get_repeated_eigenvalues(self, num: int) -> B.Numeric:
        eigenfunctions = SUEigenfunctions(self.n, num)
        eigenvalues = chain(
            eigenfunctions._eigenvalues,
            [rep_dim**2 for rep_dim in eigenfunctions._dimensions],
        )
        return B.reshape(eigenvalues, -1, 1)  # [M, 1]

    def random(self, key: B.RandomState, number: int):
        if self.n == 2:
            # explicit parametrization via the double cover SU(2) = S_3
            key, sphere_point = B.random.randn(key, dtype_double(key), number, 4)
            sphere_point /= B.reshape(
                B.sqrt(B.einsum("ij,ij->i", sphere_point, sphere_point)), -1, 1
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
            r_diag = B.einsum("...ii->...i", r)
            r_diag_inv_phase = complex_conj(
                r_diag / B.cast(B.dtype(r_diag), B.abs(r_diag))
            )
            q *= r_diag_inv_phase[:, None]
            q_det = B.det(q)
            q_det_inv_phase = complex_conj(
                (q_det / B.cast(B.dtype(q_det), B.abs(q_det)))
            )
            q_new = q[:, :, 0] * q_det_inv_phase[:, None]
            q_new = B.concat(q_new[:, :, None], q[:, :, 1:], axis=-1)
            return key, q_new

    @property
    def element_shape(self):
        """
        :return:
            [n, n].
        """
        return [self.n, self.n]

    @property
    def element_dtype(self):
        """
        :return:
            B.Complex.
        """
        return B.Complex
