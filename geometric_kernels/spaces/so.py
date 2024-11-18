"""
This module provides the :class:`SpecialOrthogonal` space, the respective
:class:`~.eigenfunctions.Eigenfunctions` subclass :class:`SOEigenfunctions`,
and a class :class:`SOCharacter` for representing characters of the group.
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
    take_along_axis,
)
from geometric_kernels.spaces.eigenfunctions import Eigenfunctions
from geometric_kernels.spaces.lie_groups import (
    CompactMatrixLieGroup,
    LieGroupCharacter,
    WeylAdditionTheorem,
)
from geometric_kernels.utils.utils import (
    chain,
    fixed_length_partitions,
    get_resource_file_path,
)


class SOEigenfunctions(WeylAdditionTheorem):
    def __init__(self, n, num_levels, compute_characters=True):
        self.n = n
        self.dim = n * (n - 1) // 2
        self.rank = n // 2

        if self.n % 2 == 0:
            self.rho = np.arange(self.rank - 1, -1, -1)
        else:
            self.rho = np.arange(self.rank - 1, -1, -1) + 0.5

        super().__init__(n, num_levels, compute_characters)

    def _generate_signatures(self, num_levels: int) -> List[Tuple[int, ...]]:
        signatures = []
        # largest LB eigenvalues correspond to partitions of smallest integers
        # IF p >> k the number of partitions of p into k parts is O(p^k)
        if self.n == 3:
            # in this case rank=1, so all partitions are trivial
            # and LB eigenvalue of the signature corresponding to p is p
            # 200 is more than enough, since the contribution of such term
            # even in Matern(0.5) case is less than 200^{-2}
            SIGNATURE_SUM = 200
        else:
            # Roughly speaking this is a bruteforce search through 50^{self.rank} smallest eigenvalues
            SIGNATURE_SUM = 50
        for signature_sum in range(0, SIGNATURE_SUM):
            for i in range(0, self.rank + 1):
                for signature in fixed_length_partitions(signature_sum, i):
                    signature.extend([0] * (self.rank - i))
                    signatures.append(tuple(signature))
                    if self.n % 2 == 0 and signature[-1] != 0:
                        signature[-1] = -signature[-1]
                        signatures.append(tuple(signature))

        eig_and_signature = [
            (round(4 * self._compute_eigenvalue(signature)), signature)
            for signature in signatures
        ]

        eig_and_signature.sort()
        signatures = [eig_sgn[1] for eig_sgn in eig_and_signature][:num_levels]
        return signatures

    def _compute_dimension(self, signature: Tuple[int, ...]) -> int:
        if self.n % 2 == 1:
            qs = [pk + self.rank - k - 1 / 2 for k, pk in enumerate(signature)]
            rep_dim = reduce(
                operator.mul,
                (2 * qs[k] / math.factorial(2 * k + 1) for k in range(0, self.rank)),
            ) * reduce(
                operator.mul,
                (
                    (qs[i] - qs[j]) * (qs[i] + qs[j])
                    for i, j in itertools.combinations(range(self.rank), 2)
                ),
                1,
            )
            return int(round(rep_dim))
        else:
            qs = [
                pk + self.rank - k - 1 if k != self.rank - 1 else abs(pk)
                for k, pk in enumerate(signature)
            ]
            rep_dim = int(
                reduce(
                    operator.mul,
                    (2 / math.factorial(2 * k) for k in range(1, self.rank)),
                )
                * reduce(
                    operator.mul,
                    (
                        (qs[i] - qs[j]) * (qs[i] + qs[j])
                        for i, j in itertools.combinations(range(self.rank), 2)
                    ),
                    1,
                )
            )
            return int(round(rep_dim))

    def _compute_eigenvalue(self, signature: Tuple[int, ...]) -> B.Float:
        np_sgn = np.array(signature)
        rho = self.rho
        eigenvalue = np.linalg.norm(rho + np_sgn) ** 2 - np.linalg.norm(rho) ** 2
        return eigenvalue

    def _compute_character(
        self, n: int, signature: Tuple[int, ...]
    ) -> LieGroupCharacter:
        return SOCharacter(n, signature)

    def _torus_representative(self, X: B.Numeric) -> B.Numeric:
        gamma = None

        if self.n == 3:
            # In SO(3) the torus representative is determined by the non-trivial pair of eigenvalues,
            # which can be calculated from the trace
            trace = B.expand_dims(B.trace(X), axis=-1)
            real = (trace - 1) / 2
            zeros = real * 0
            imag = B.sqrt(B.maximum(1 - real * real, zeros))
            gamma = create_complex(real, imag)
        elif self.n % 2 == 1:
            # In SO(2n+1) the torus representative is determined by the (unordered) non-trivial eigenvalues
            eigvals = B.eig(X, False)
            sorted_ind = B.argsort(B.real(eigvals), axis=-1)
            eigvals = take_along_axis(eigvals, sorted_ind, -1)
            gamma = eigvals[..., 0:-1:2]
        else:
            # In SO(2n) each unordered set of eigenvalues determines two conjugacy classes
            eigvals, eigvecs = B.eig(X)
            sorted_ind = B.argsort(B.real(eigvals), axis=-1)
            eigvals = take_along_axis(eigvals, sorted_ind, -1)
            eigvecs = take_along_axis(
                eigvecs,
                B.broadcast_to(B.expand_dims(sorted_ind, axis=-2), *eigvecs.shape),
                -1,
            )
            # c is a matrix transforming x into its canonical form (with 2x2 blocks)
            c = B.reshape(
                B.stack(
                    eigvecs[..., ::2] + eigvecs[..., 1::2],
                    eigvecs[..., ::2] - eigvecs[..., 1::2],
                    axis=-1,
                ),
                *eigvecs.shape[:-1],
                -1,
            )
            # eigenvectors calculated by LAPACK are either real or purely imaginary, make everything real
            # WARNING: might depend on the implementation of the eigendecomposition!
            c = B.real(c) + B.imag(c)
            # normalize s.t. det(c)≈±1, probably unnecessary
            c /= math.sqrt(2)
            eigvals = B.concat(
                B.expand_dims(
                    B.power(eigvals[..., 0], B.cast(complex_like(c), B.sign(B.det(c)))),
                    axis=-1,
                ),
                eigvals[..., 1:],
                axis=-1,
            )
            gamma = eigvals[..., ::2]
        gamma = B.concat(gamma, complex_conj(gamma), axis=-1)
        return gamma

    def inverse(self, X: B.Numeric) -> B.Numeric:
        return SpecialOrthogonal.inverse(X)


class SOCharacter(LieGroupCharacter):
    """
    The class that represents a character of the SO(n) group.

    These characters are always real-valued.

    These are polynomials whose coefficients are precomputed and stored in a
    file. By default, there are 20 precomputed characters for n from 3 to 8.
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
        group_name = "SO({})".format(self.n)
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
                gammas ** B.cast(B.dtype(gammas), from_numpy(gammas, monom)), axis=-1
            )
        return char_val


class SpecialOrthogonal(CompactMatrixLieGroup):
    r"""
    The GeometricKernels space representing the special orthogonal group SO(n)
    consisting of n by n orthogonal matrices with unit determinant.

    The elements of this space are represented as n x n orthogonal
    matrices with real entries and unit determinant.

    .. note::
        A tutorial on how to use this space is available in the
        :doc:`SpecialOrthogonal.ipynb </examples/SpecialOrthogonal>` notebook.

    :param n:
        The order n of the group SO(n).

    .. note::
        We only support n >= 3. Mathematically, SO(2) is equivalent to the
        unit circle, which is available as the :class:`~.spaces.Circle` space.

        For larger values of n, you might need to run the
        `utils/compute_characters.py` script to precompute the necessary
        mathematical quantities beyond the ones provided by default. Same
        can be required for larger numbers of levels.

    .. admonition:: Citation

        If you use this GeometricKernels space in your research, please consider
        citing :cite:t:`azangulov2024a`.
    """

    def __init__(self, n: int):
        if n < 3:
            raise ValueError("Only n >= 3 is supported. For n = 2, use Circle.")
        self.n = n
        self.dim = n * (n - 1) // 2
        self.rank = n // 2
        super().__init__()

    def __str__(self):
        return f"SpecialOrthogonal({self.n})"

    @property
    def dimension(self) -> int:
        """
        The dimension of the space, as that of a Riemannian manifold.

        :return:
            floor(n(n-1)/2) where n is the order of the group SO(n).
        """
        return self.dim

    @staticmethod
    def inverse(X: B.Numeric) -> B.Numeric:
        return B.transpose(X)  # B.transpose only inverses the last two dims.

    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        Returns the :class:`~.SOEigenfunctions` object with `num` levels
        and order n.

        :param num:
            Number of levels.
        """
        return SOEigenfunctions(self.n, num)

    def get_eigenvalues(self, num: int) -> B.Numeric:
        eigenfunctions = SOEigenfunctions(self.n, num)
        eigenvalues = np.array(
            [eigenvalue for eigenvalue in eigenfunctions._eigenvalues]
        )
        return B.reshape(eigenvalues, -1, 1)  # [num, 1]

    def get_repeated_eigenvalues(self, num: int) -> B.Numeric:
        eigenfunctions = SOEigenfunctions(self.n, num)
        eigenvalues = chain(
            eigenfunctions._eigenvalues,
            [rep_dim**2 for rep_dim in eigenfunctions._dimensions],
        )
        return B.reshape(eigenvalues, -1, 1)  # [J, 1]

    def random(self, key: B.RandomState, number: int):
        if self.n == 2:  # for the bright future where we support SO(2).
            # SO(2) = S^1
            key, thetas = B.random.randn(key, dtype_double(key), number, 1)
            thetas = 2 * math.pi * thetas
            c = B.cos(thetas)
            s = B.sin(thetas)
            r1 = B.stack(c, s, axis=-1)
            r2 = B.stack(-s, c, axis=-1)
            q = B.concat(r1, r2, axis=-2)
            return key, q
        elif self.n == 3:
            # explicit parametrization via the double cover SU(2) = S^3
            key, sphere_point = B.random.randn(key, dtype_double(key), number, 4)
            sphere_point /= B.reshape(
                B.sqrt(B.einsum("ij,ij->i", sphere_point, sphere_point)), -1, 1
            )

            x, y, z, w = (B.reshape(sphere_point[..., i], -1, 1) for i in range(4))
            xx, yy, zz = x**2, y**2, z**2
            xy, xz, xw, yz, yw, zw = x * y, x * z, x * w, y * z, y * w, z * w

            r1 = B.stack(1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw), axis=1)
            r2 = B.stack(2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw), axis=1)
            r3 = B.stack(2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy), axis=1)

            q = B.concat(r1, r2, r3, axis=-1)
            return key, q
        else:
            # qr decomposition is not in the lab package, so numpy is used.
            key, h = B.random.randn(key, dtype_double(key), number, self.n, self.n)
            q, r = qr(h, mode="complete")
            r_diag_sign = B.sign(B.einsum("...ii->...i", r))
            q *= r_diag_sign[:, None]
            q_det_sign = B.sign(B.det(q))
            q_new = q[:, :, 0] * q_det_sign[:, None]
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
            B.Float.
        """
        return B.Float
