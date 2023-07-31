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
from functools import reduce
from sympy.matrices.determinant import _det as sp_det
from geometric_kernels.lab_extras import dtype_double, from_numpy, qr, take_along_axis
from geometric_kernels.utils.utils import fixed_length_partitions, partition_dominance_or_subpartition_cone
from geometric_kernels.spaces.lie_groups import LieGroup, LieGroupAddtitionTheorem, LieGroupCharacter


class SOEigenfunctions(LieGroupAddtitionTheorem):
    def __init__(self, n, num_levels,  init_eigenfunctions=True):
        self.n = n
        self.dim = n * (n-1) // 2
        self.rank = n // 2

        if self.n % 2 == 0:
            self.rho = np.arange(self.rank - 1, -1, -1)
        else:
            self.rho = np.arange(self.rank - 1, -1, -1) + 0.5

        self._num_levels = num_levels
        self._signatures = self._generate_signatures(self._num_levels)
        self._eigenvalues = np.array([self._compute_eigenvalue(signature) for signature in self._signatures])
        self._dimensions = np.array([self._compute_dimension(signature) for signature in self._signatures])

        if init_eigenfunctions:
            self._characters = [SOCharacter(n, signature) for signature in self._signatures]

    def _generate_signatures(self, num_levels):
        """
        Generate the signatures of irreducible representations
        Representations of SO(dim) can be enumerated by partitions of size rank, called signatures.
        :return signatures: list of signatures of representations likely having the smallest LB eigenvalues
        """
        signatures = []
        # largest LB eigenvalues correspond to partitions of smallest integers
        # IF p >> k the number of partitions of p into k parts is O(p^k)
        if self.n == 3:
            # in this case rank=1, so all partitions are trivial
            # and LB eigenvalue of signature corresponding to p is p
            # 200 is more than enough, since the contribution of such term
            # even in Matern(0.5) case is less than 200^{-2}
            signature_sum = 200
        else:
            # Roughly speaking this is a bruteforce search through 50^{self.rank} smallest eigenvalues
            signature_sum = 50
        for signature_sum in range(0, signature_sum):
            for i in range(0, self.rank + 1):
                for signature in fixed_length_partitions(signature_sum, i):
                    signature.extend([0] * (self.rank - i))
                    signatures.append(tuple(signature))
                    if self.n % 2 == 0 and signature[-1] != 0:
                        signature[-1] = -signature[-1]
                        signatures.append(tuple(signature))

        dimensions = [self._compute_eigenvalue(signature) for signature in signatures]
        min_ind = np.argpartition(dimensions, num_levels)[:num_levels]
        signatures = [signatures[i] for i in min_ind]
        return signatures

    def _compute_dimension(self, signature):
        if self.n % 2 == 1:
            qs = [pk + self.rank - k - 1 / 2 for k, pk in enumerate(signature)]
            rep_dim = reduce(operator.mul, (2 * qs[k] / math.factorial(2 * k + 1) for k in range(0, self.rank))) \
                      * reduce(operator.mul, ((qs[i] - qs[j]) * (qs[i] + qs[j])
                                              for i, j in itertools.combinations(range(self.rank), 2)), 1)
            return int(round(rep_dim))
        else:
            qs = [pk + self.rank - k - 1 if k != self.rank - 1 else abs(pk) for k, pk in enumerate(signature)]
            rep_dim = int(reduce(operator.mul, (2 / math.factorial(2 * k) for k in range(1, self.rank)))
                          * reduce(operator.mul, ((qs[i] - qs[j]) * (qs[i] + qs[j])
                                                  for i, j in itertools.combinations(range(self.rank), 2)), 1))
            return int(round(rep_dim))

    def _compute_eigenvalue(self, signature):
        np_sgn = np.array(signature)
        rho = self.rho
        eigenvalue = (np.linalg.norm(rho + np_sgn) ** 2 - np.linalg.norm(rho) ** 2)
        return eigenvalue.item()

    def _torus_representative(self, X):
            """
            The function maps Lie Group Element X to T -- a maximal torus of the Lie group
            [n1,n2,\ldots, nk,X, X] ---> [n1,n2,\ldots,nk,X, X]
            """

            gamma = None

            if self.n == 3:
                # In SO(3) the torus representative is determined by the non-trivial pair of eigenvalues,
                # which can be calculated from the trace
                trace = B.expand_dims(B.trace(X), axis=-1)
                real = (trace - 1) / 2
                zeros = real * 0
                imag = B.sqrt(B.maximum(1 - real*real, zeros))
                gamma = real + 1j*imag
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
                eigvecs = take_along_axis(eigvecs, B.broadcast_to(B.expand_dims(sorted_ind, axis=-2), *eigvecs.shape), -1)
                # c is a matrix transforming x into its canonical form (with 2x2 blocks)
                c = 0*eigvecs
                c[..., ::2] = eigvecs[..., ::2] + eigvecs[..., 1::2]
                c[..., 1::2] = (eigvecs[..., ::2] - eigvecs[..., 1::2])
                # eigenvectors calculated by LAPACK are either real or purely imaginary, make everything real
                # WARNING: might depend on the implementation of the eigendecomposition!
                c = c.real + c.imag
                # normalize s.t. det(c)≈±1, probably unnecessary
                c /= math.sqrt(2)
                eigvals[..., 0] = B.power(eigvals[..., 0], B.sign(B.det(c)))
                gamma = eigvals[..., ::2]
            gamma = B.concat(gamma, gamma.conj(), axis=-1)
            return gamma

    def _compute_character_formula(self, signature):
        n = self.n
        rank = self.rank
        gammas = sympy.symbols(' '.join('g{}'.format(i + 1) for i in range(rank)))
        gammas = list(more_itertools.always_iterable(gammas))
        gammas_conj = sympy.symbols(' '.join('gc{}'.format(i + 1) for i in range(rank)))
        gammas_conj = list(more_itertools.always_iterable(gammas_conj))
        chi_variables = gammas + gammas_conj
        if n % 2:
            gammas_sqrt = sympy.symbols(' '.join('gr{}'.format(i + 1) for i in range(rank)))
            gammas_sqrt = list(more_itertools.always_iterable(gammas_sqrt))
            gammas_conj_sqrt = sympy.symbols(' '.join('gcr{}'.format(i + 1) for i in range(rank)))
            gammas_conj_sqrt = list(more_itertools.always_iterable(gammas_conj_sqrt))
            chi_variables = gammas_sqrt + gammas_conj_sqrt
            def xi1(qs):
                mat = sympy.Matrix(rank, rank, lambda i, j: gammas_sqrt[i]**qs[j]-gammas_conj_sqrt[i]**qs[j])
                return sympy.Poly(sp_det(mat, method='berkowitz'), chi_variables)
            # qs = [sympy.Integer(2*pk + 2*rank - 2*k - 1) / 2 for k, pk in enumerate(signature)]
            qs = [2 * pk + 2 * rank - 2 * k - 1 for k, pk in enumerate(signature)]
            # denom_pows = [sympy.Integer(2*k - 1) / 2 for k in range(rank, 0, -1)]
            denom_pows = [2 * k - 1 for k in range(rank, 0, -1)]
            numer = xi1(qs)
            denom = xi1(denom_pows)
        else:
            def xi0(qs):
                mat = sympy.Matrix(rank, rank, lambda i, j: gammas[i] ** qs[j] + gammas_conj[i] ** qs[j])
                return sympy.Poly(sp_det(mat, method='berkowitz'), chi_variables)
            def xi1(qs):
                mat = sympy.Matrix(rank, rank, lambda i, j: gammas[i] ** qs[j] - gammas_conj[i] ** qs[j])
                return sympy.Poly(sp_det(mat, method='berkowitz'), chi_variables)
            qs = [pk + rank - k - 1 if k != rank - 1 else abs(pk) for k, pk in enumerate(signature)]
            pm = signature[-1]
            numer = xi0(qs)
            if pm:
                numer += (1 if pm > 0 else -1) * xi1(qs)
            denom = xi0(list(reversed(range(rank))))
        partition = tuple(map(abs, self.representation.index)) + tuple([0] * self.representation.manifold.rank)
        monomials_tuples = itertools.chain.from_iterable(
            more_itertools.distinct_permutations(p) for p in partition_dominance_or_subpartition_cone(partition)
        )
        monomials_tuples = filter(lambda p: all(p[i] == 0 or p[i + rank] == 0 for i in range(rank)), monomials_tuples)
        monomials_tuples = list(monomials_tuples)
        monomials = [sympy.polys.monomials.Monomial(m, chi_variables).as_expr()
                     for m in monomials_tuples]
        chi_coeffs = list(more_itertools.always_iterable(
            sympy.symbols(' '.join('c{}'.format(i) for i in range(1, len(monomials) + 1)))))
        exponents = [n % 2 + 1] * len(monomials)  # the correction s.t. chi is the same polynomial for both oddities of n
        chi_poly = sympy.Poly(sum(c * m**d for c, m, d in zip(chi_coeffs, monomials, exponents)), chi_variables)
        pr = chi_poly * denom - numer
        if n % 2:
            pr = sympy.Poly(pr.subs((g*gc, 1) for g, gc in zip(gammas_sqrt, gammas_conj_sqrt)), chi_variables)
        else:
            pr = sympy.Poly(pr.subs((g*gc, 1) for g, gc in zip(gammas, gammas_conj)), chi_variables)
        sol = list(sympy.linsolve(pr.coeffs(), chi_coeffs)).pop()
        if n % 2:
            chi_variables = gammas + gammas_conj
            chi_poly = sympy.Poly(chi_poly.subs([gr ** 2, g] for gr, g in zip(gammas_sqrt + gammas_conj_sqrt, chi_variables)), chi_variables)
        p = sympy.Poly(chi_poly.subs((c, c_val) for c, c_val in zip(chi_coeffs, sol)), chi_variables)
        coeffs = list(map(int, p.coeffs()))
        monoms = [list(map(int, monom)) for monom in p.monoms()]
        return coeffs, monoms

    def inverse(self, X: B.Numeric) -> B.Numeric:
        return B.transpose(X)

    def num_levels(self) -> int:
            """Number of levels, L"""
            return self._num_levels

    def num_eigenfunctions_per_level(self) -> int:
        """Number of eigenfunctions per level"""
        return self._dimensions


class SOCharacter(LieGroupCharacter):
    def __init__(self, n, signature):
        self.signature = signature
        self.n = n
        self.coeffs, self.monoms = self._load()

    def _load(self):
        group_name = 'SO({})'.format(self.n)
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
                               'Run compute_characters.py with changed parameters.'.format(e.args[0], group_name)) from None

    def __call__(self, gammas):
        char_val = B.zeros(B.dtype(gammas), *gammas.shape[:-1])
        for coeff, monom in zip(self.coeffs, self.monoms):
            char_val += coeff * B.prod(gammas ** from_numpy(gammas, monom), axis=-1)
        return char_val


class SOGroup(LieGroup):
    r"""
    The d-dimensional hypersphere embedded in the (d+1)-dimensional Euclidean space.
    """
    def __init__(self, n):
        self.n = n
        self.dim = n * (n-1) // 2
        self.rank = n // 2
        LieGroup.__init__(self)


    @property
    def dimension(self) -> int:
        return self.dim

    def inverse(self, X: B.Numeric) -> B.Numeric:
        return B.transpose(X)

    def get_eigenfunctions(self, num: int) -> SOEigenfunctions:
        """
        :param num: number of eigenfunctions returned.
        """
        return SOEigenfunctions(self.n, num)

    def get_eigenvalues(self, num: int) -> B.Numeric:
        """
        First `num` eigenvalues of the Laplace-Beltrami operator

        :return: [num, 1] array containing the eigenvalues
        """
        eigenfunctions = SOEigenfunctions(self.n, num)
        eigenvalues = np.array(
            [eigenvalue for eigenvalue in eigenfunctions._eigenvalues]
        )
        return B.reshape(eigenvalues, -1, 1)  # [num, 1]

    def get_repeated_eigenvalues(self, num: int) -> B.Numeric:
        """Eigenvalues of first 'num' levels of the Laplace-Beltrami operator,
        repeated according to their multiplicity.

        :return: [M, 1] array containing the eigenvalues
        """
        eigenfunctions = SOEigenfunctions(self.n, num)
        eigenvalues = np.array(
            itertools.chain([
                [eigenfunction] * dim
                for eigenfunction, dim in zip(eigenfunctions._eigenvalues, eigenfunctions._dimensions)
            ])
        )
        return B.reshape(eigenvalues, -1, 1)  # [M, 1]

    def random(self, key, number):
        if self.n == 2:
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
            sphere_point /= B.reshape(B.sqrt(einsum('ij,ij->i', sphere_point, sphere_point)), -1, 1)

            x, y, z, w = (B.reshape(sphere_point[..., i], -1, 1) for i in range(4))
            xx, yy, zz = x ** 2, y ** 2, z ** 2
            xy, xz, xw, yz, yw, zw = x * y, x * z, x * w, y * z, y * w, z * w

            r1 = B.stack(1-2*(yy+zz), 2*(xy-zw), 2*(xz+yw), axis=1)
            r2 = B.stack(2*(xy+zw), 1-2*(xx+zz), 2*(yz-xw), axis=1)
            r3 = B.stack(2*(xz-yw), 2*(yz+xw), 1-2*(xx+yy), axis=1)

            q = B.concat(r1, r2, r3, axis=-1)
            return key, q
        else:
            # qr decomposition is not in the lab package, so numpy is used.
            key, h = B.random.randn(key, dtype_double(key), number, self.n, self.n)
            q, r = qr(h)
            r_diag_sign = B.sign(einsum('...ii->...i', r))
            q *= r_diag_sign[:, None]
            q_det_sign = B.sign(B.det(q))
            q[:, :, 0] *= q_det_sign[:, None]
            return key, q
