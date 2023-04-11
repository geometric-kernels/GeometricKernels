import json
import math
import operator
import itertools
import more_itertools
import sympy
import lab as B
import numpy as np
from opt_einsum import contract as einsum
from pathlib import Path
from sympy.matrices.determinant import _det as sp_det
from functools import reduce
from geometric_kernels.lab_extras import dtype_double, from_numpy, qr
from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import EigenfunctionWithAdditionTheorem, Eigenfunctions
from geometric_kernels.utils.utils import fixed_length_partitions, partition_dominance_cone, partition_dominance_or_subpartition_cone

from geometric_kernels.spaces.lie_groups import LieGroup, LieGroupAddtitionTheorem, LieGroupCharacter


class SUEigenfunctions(LieGroupAddtitionTheorem):
    def __init__(self, n, num_levels, init_eigenfunctions=True):
        self.n = n
        self.dim = n * (n-1)
        self.rank = n - 1

        self.rho = np.arange(self.n - 1, -self.n, -2) * 0.5

        self._num_levels = num_levels
        self._signatures = self._generate_signatures(self._num_levels)
        self._eigenvalues = np.array([self._compute_eigenvalue(signature) for signature in self._signatures])
        self._dimensions = np.array([self._compute_dimension(signature) for signature in self._signatures])
        if init_eigenfunctions:
            self._characters = [SUCharacter(n, signature) for signature in self._signatures]



    def _generate_signatures(self, num_levels):
        """
        Generate the signatures of irreducible representations
        Representations of SU(dim) can be enumerated by partitions of size dim, called signatures.
        :param int order: number of eigenfunctions that will be returned
        :return signatures: signatures of representations likely having the smallest LB eigenvalues
        """

        sign_vals_lim = 100 if self.n in (1, 2) else 30 if self.n == 3 else 10
        signatures = list(itertools.combinations_with_replacement(range(sign_vals_lim, -1, -1), r=self.rank))
        signatures = [sgn + (0,) for sgn in signatures]

        dimensions = [self._compute_eigenvalue(signature) for signature in signatures]
        min_ind = np.argpartition(dimensions, num_levels)[:num_levels]
        signatures = [signatures[i] for i in min_ind]
        return signatures

    def _compute_dimension(self, signature):
        rep_dim = reduce(operator.mul, (reduce(operator.mul, (signature[i - 1] - signature[j - 1] + j - i for j in
                                                              range(i + 1, self.n + 1))) / math.factorial(self.n - i)
                                        for i in range(1, self.n)))
        return int(round(rep_dim))

    def _compute_eigenvalue(self, signature):
        normalized_signature = signature - np.mean(signature)
        lb_eigenvalue = (np.linalg.norm(self.rho + normalized_signature) ** 2 - np.linalg.norm(self.rho) ** 2)
        return lb_eigenvalue.item()


    def _torus_representative(self, X):
        return B.eig(X, False)

    def _compute_character_formula(self):
        n = self.representation.manifold.n
        gammas = sympy.symbols(' '.join('g{}'.format(i) for i in range(1, n + 1)))
        qs = [pk + n - k - 1 for k, pk in enumerate(self.representation.index)]
        numer_mat = sympy.Matrix(n, n, lambda i, j: gammas[i]**qs[j])
        numer = sympy.Poly(sp_det(numer_mat, method='berkowitz'))
        denom = sympy.Poly(sympy.prod(gammas[i] - gammas[j] for i, j in itertools.combinations(range(n), r=2)))
        monomials_tuples = list(itertools.chain.from_iterable(
            more_itertools.distinct_permutations(p) for p in partition_dominance_cone(self.representation.index)
        ))
        monomials = [sympy.polys.monomials.Monomial(m, gammas).as_expr() for m in monomials_tuples]
        chi_coeffs = list(more_itertools.always_iterable(sympy.symbols(' '.join('c{}'.format(i) for i in range(1, len(monomials) + 1)))))
        chi_poly = sympy.Poly(sum(c * m for c, m in zip(chi_coeffs, monomials)), gammas)
        pr = chi_poly * denom - numer
        sol = list(sympy.linsolve(pr.coeffs(), chi_coeffs)).pop()
        p = sympy.Poly(sum(c * m for c, m in zip(sol, monomials)), gammas)
        coeffs = list(map(int, p.coeffs()))
        monoms = [list(map(int, monom)) for monom in p.monoms()]
        return coeffs, monoms

    def inverse(self, X: B.Numeric) -> B.Numeric:
        return B.transpose(X).conj()

    def num_levels(self) -> int:
            """Number of levels, L"""
            return self._num_levels

    def num_eigenfunctions_per_level(self) -> int:
        """Number of eigenfunctions per level"""
        return self._dimensions


class SUCharacter(LieGroupCharacter):
    def __init__(self, n, signature):
        self.signature = signature
        self.n = n
        self.coeffs, self.monoms = self._load()

    def _load(self):
        group_name = 'SU({})'.format(self.n)
        file_path = Path(__file__).with_name('precomputed_characters.json')
        with file_path.open('r') as file:
            character_formulas = json.load(file)
            try:
                cs, ms = character_formulas[group_name][str(self.signature)]
                coeffs, monoms = (np.array(data) for data in (cs, ms))
                return coeffs, monoms
            except KeyError as e:
                raise KeyError('Unable to retrieve character parameters for signature {} of {}, '
                               'perhaps it is not precomputed.'
                               'Run compute_characters.py with changed parameters.'.format(e.args[0],
                                                                                           group_name)) from None


    def __call__(self, gammas):
            char_val = B.zeros(B.dtype(gammas), *gammas.shape[:-1])
            for coeff, monom in zip(self.coeffs, self.monoms):
                char_val += coeff * B.prod(gammas ** from_numpy(gammas, monom), axis=-1)
            return char_val


class SUGroup(LieGroup):
    r"""
    The d-dimensional hypersphere embedded in the (d+1)-dimensional Euclidean space.
    """
    def __init__(self, n):
        self.n = n
        self.dim = n * (n-1) // 2
        LieGroup.__init__(self)

    @property
    def dimension(self) -> int:
        return self.dim

    def inverse(self, X: B.Numeric) -> B.Numeric:
        return B.transpose(X).conj()


    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        :param num: number of eigenfunctions returned.
        """
        return SUEigenfunctions(self.n, num)

    def get_eigenvalues(self, num: int) -> B.Numeric:
        """
        First `num` eigenvalues of the Laplace-Beltrami operator

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
        eigenvalues = np.array(
            itertools.chain([
                [eigenfunction] * dim
                for eigenfunction, dim in zip(eigenfunctions._eigenvalues, eigenfunctions._dimensions)
            ])
        )
        return B.reshape(eigenvalues, -1, 1)  # [M, 1]

    def random(self, key, number):
        if self.n == 2:
            # explicit parametrization via the double cover SU(2) = S^3
            key, sphere_point = B.random.randn(key, dtype_double(key), number, 4)
            sphere_point /= B.reshape(B.sqrt(einsum('ij,ij->i', sphere_point, sphere_point)), -1, 1)
            a = sphere_point[..., 0] + 1j*sphere_point[..., 1]
            b = sphere_point[..., 2] + 1j*sphere_point[..., 3]
            r1 = B.stack(a, -b.conj(), axis=-1)
            r2 = B.stack(b, a.conj(), axis=-1)
            q = B.stack(r1, r2, axis=-1)
            return key, q
        else:
            key, real = B.random.randn(key, dtype_double(key), number, self.n, self.n)
            key, imag = B.random.randn(key, dtype_double(key), number, self.n, self.n)
            h = (real +1j*imag)/math.sqrt(2)
            q, r = qr(h)
            r_diag = einsum('...ii->...i', r)
            r_diag_inv_phase = (r_diag / B.abs(r_diag)).conj()
            q *= r_diag_inv_phase[:, None]
            q_det = B.det(q)
            q_det_inv_phase = (q_det / B.abs(q_det)).conj()
            q[:, :, 0] *= q_det_inv_phase[:, None]
            return key, q