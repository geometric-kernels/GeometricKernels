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
from geometric_kernels.lab_extras import dtype_double, from_numpy
from geometric_kernels.spaces.base import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import EigenfunctionWithAdditionTheorem, Eigenfunctions


from geometric_kernels.spaces.lie_spaces import LieGroup, LieGroupAddtitionTheorem, LieGroupCharacter
import geomstats as gs



def fixed_length_partitions(n, L):
    """
    https://www.ics.uci.edu/~eppstein/PADS/IntegerPartitions.py
    Integer partitions of n into L parts, in colex order.
    The algorithm follows Knuth v4 fasc3 p38 in rough outline;
    Knuth credits it
     to Hindenburg, 1779.
    """

    # guard against special cases
    if L == 0:
        if n == 0:
            yield []
        return
    if L == 1:
        if n > 0:
            yield [n]
        return
    if n < L:
        return

    partition = [n - L + 1] + (L - 1) * [1]
    while True:
        yield partition.copy()
        if partition[0] - 1 > partition[1]:
            partition[0] -= 1
            partition[1] += 1
            continue
        j = 2
        s = partition[0] + partition[1] - 1
        while j < L and partition[j] >= partition[0] - 1:
            s += partition[j]
            j += 1
        if j >= L:
            return
        partition[j] = x = partition[j] + 1
        j -= 1
        while j > 0:
            partition[j] = x
            s -= x
            j -= 1
        partition[0] = s


def partition_dominance_cone(partition):
    '''
    Calculates partitions dominated by a given one
    and having the same number of parts (including zero parts of the original)
    '''
    cone = {partition}
    new_partitions = {0}
    prev_partitions = cone
    while new_partitions:
        new_partitions = set()
        for partition in prev_partitions:
            for i in range(len(partition) - 1):
                if partition[i] > partition[i + 1]:
                    for j in range(i + 1, len(partition)):
                        if partition[i] > partition[j] + 1 and partition[j] < partition[j - 1]:
                            new_partition = list(partition)
                            new_partition[i] -= 1
                            new_partition[j] += 1
                            new_partition = tuple(new_partition)
                            if new_partition not in cone:
                                new_partitions.add(new_partition)
        cone.update(new_partitions)
        prev_partitions = new_partitions
    return cone


def partition_dominance_or_subpartition_cone(partition):
    '''
        Calculates subpartitions and partitions dominated by a given one
        and having the same number of parts (including zero parts of the original)
        '''
    cone = {partition}
    new_partitions = {0}
    prev_partitions = cone
    while new_partitions:
        new_partitions = set()
        for partition in prev_partitions:
            for i in range(len(partition) - 1):
                if partition[i] > partition[i + 1]:
                    new_partition = list(partition)
                    new_partition[i] -= 1
                    new_partition = tuple(new_partition)
                    if new_partition not in cone:
                        new_partitions.add(new_partition)
                    for j in range(i + 1, len(partition)):
                        if partition[i] > partition[j] + 1 and partition[j] < partition[j - 1]:
                            new_partition = list(partition)
                            new_partition[i] -= 1
                            new_partition[j] += 1
                            new_partition = tuple(new_partition)
                            if new_partition not in cone:
                                new_partitions.add(new_partition)
        cone.update(new_partitions)
        prev_partitions = new_partitions
    return cone


class SOEigenfunction(LieGroupAddtitionTheorem):
    def __init__(self, num_levels, n, init_eigenfunctions=True):
        self.n = n
        self.dim = n * (n-1) // 2
        self.rank = n // 2

        if self.n % 2 == 0:
            self.rho = np.arange(self.rank - 1, -1, -1)
        else:
            self.rho = np.arange(self.rank - 1, -1, -1) + 0.5

        self._num_levels = num_levels
        self._signatures = self.generate_signatures(self._num_levels) # both have the length num_levels
        self._eigenvalues = np.array([self.compute_eigenvalue(signature) for signature in self._signatures])
        self._dimensions = np.array([self.compute_dimension(signature) for signature in self._signatures])
        if init_eigenfunctions:
            self._characters = [SOCharacter(n, signature) for signature in self._signatures]



    def _generate_signatures(self):
        """Generate the signatures of irreducible representations
                Representations of SO(dim) can be enumerated by partitions of size dim, called signatures.
                :param int order: number of eigenfunctions that will be returned
                :return signatures: signatures of representations likely having the smallest LB eigenvalues
                """
        signatures = []
        if self.n == 3:
            signature_sum = 200
        else:
            signature_sum = 30
        for signature_sum in range(0, signature_sum):
            for i in range(0, self.rank + 1):
                for signature in fixed_length_partitions(signature_sum, i):
                    signature.extend([0] * (self.rank - i))
                    signatures.append(tuple(signature))
                    if self.n % 2 == 0 and signature[-1] != 0:
                        signature[-1] = -signature[-1]
                        signatures.append(tuple(signature))
        return signatures

    def compute_dimension(self, signature):
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
        eigenvalue = (np.linalg.norm(rho + np_sgn) ** 2 - np.linalg.norm(rho) ** 2)  # / killing_form_coeff
        return eigenvalue.item()

    def _torus_representative(self, X):
            """The function maps Lie Group Element X to T -- a maximal torus of the Lie group
            [n1,n2,\ldots, nk,X, X] ---> [n1,n2,\ldots,nk,X, X]"""
            if self.n == 3:
                # In SO(3) the torus representative is determined by the non-trivial pair of eigenvalues,
                # which can be calculated from the trace
                trace = einsum('...ii->...', X)
                real = (trace - 1) / 2
                imag = B.sqrt(B.max(1 - B.square(real), B.zeros_like(real)))
                return B.view_as_complex(B.cat((real.unsqueeze(-1), imag.unsqueeze(-1)), -1)).unsqueeze(-1)
            elif self.n % 2 == 1:
                # In SO(2n+1) the torus representative is determined by the (unordered) non-trivial eigenvalues
                eigvals = B.linalg.eigvals(X)
                sorted_ind = B.sort(B.view_as_real(eigvals), dim=-2).indices[..., 0]
                eigvals = B.gather(eigvals, dim=-1, index=sorted_ind)
                gamma = eigvals[..., 0:-1:2]
                return gamma
            else:
                # In SO(2n) each unordered set of eigenvalues determines two conjugacy classes
                eigvals, eigvecs = B.linalg.eig(X)
                sorted_ind = B.sort(B.view_as_real(eigvals), dim=-2).indices[..., 0]
                eigvals = B.gather(eigvals, dim=-1, index=sorted_ind)
                eigvecs = B.gather(eigvecs, dim=-1, index=sorted_ind.unsqueeze(-2).broadcast_to(eigvecs.shape))
                # c is a matrix transforming x into its canonical form (with 2x2 blocks)
                c = B.zeros_like(eigvecs)
                c[..., ::2] = eigvecs[..., ::2] + eigvecs[..., 1::2]
                c[..., 1::2] = (eigvecs[..., ::2] - eigvecs[..., 1::2])
                # eigenvectors calculated by LAPACK are either real or purely imaginary, make everything real
                # WARNING: might depend on the implementation of the eigendecomposition!
                c = c.real + c.imag
                # normalize s.t. det(c)≈±1, probably unnecessary
                c /= math.sqrt(2)
                B.pow(eigvals[..., 0], B.det(c).sgn(), out=eigvals[..., 0])
                gamma = eigvals[..., ::2]
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

    def _inverse(self, X: B.Numeric) -> B.Numeric:
        return B.inverse(X)

    def num_levels(self) -> int:
            """Number of levels, L"""
            return self._num_levels

    def num_eigenfunctions_per_level(self) -> int:
        """Number of eigenfunctions per level
        Is it right?"""
        return self._dimensions


class SOCharacter(LieGroupCharacter):
    def __init__(self, n, signature):
        self.signature = signature
        self.n = n
        self.coeffs, self.monoms = self._load()

    def _load(self):
        group_name = '{SO}({})'.format(self.n)
        file_path = Path(__file__).with_name('precomputed_characters.json')
        with file_path.open('r') as file:
            character_formulas = json.load(file)
            try:
                cs, ms = character_formulas[group_name][str(self.signature)]
                self.coeffs, self.monoms = (np.array(data) for data in (cs, ms))
            except KeyError as e:
                raise KeyError('Unable to retrieve character parameters for signature {} of {}, '
                               'perhaps it is not precomputed.'.format(e.args[0], group_name)) from None

    def __call__(self, gammas):
        gammas = B.cat((gammas, gammas.conj()), dim=-1)
        char_val = B.zeros(B.dtype(gammas), gammas.shape[:-1])
        for coeff, monom in zip(self.coeffs, self.monoms):
            char_val += coeff * B.prod(gammas ** from_numpy(gammas, monom), dim=-1)
        return char_val


class SOGroup(LieGroup, gs.geometry.special_orthogonal):
    r"""
    The d-dimensional hypersphere embedded in the (d+1)-dimensional Euclidean space.
    """
    def __init__(self, n):
        self.n = n
        self.dim = n * (n-1) // 2

    def random(self, key, number):
        if self.n == 2:
            # SO(2) = S^1
            thetas = 2 * math.pi * B.random.randn(key, dtype_double(key), number, 1)
            c = B.cos(thetas)
            s = B.sin(thetas)
            r1 = B.hstack((c, s)).unsqueeze(-2)
            r2 = B.hstack((-s, c)).unsqueeze(-2)
            q = B.cat((r1, r2), dim=-2)
            return q
        elif self.n == 3:
            # explicit parametrization via the double cover SU(2) = S^3
            sphere_point = B.random.randn(key, dtype_double(key), number, 4)
            sphere_point /= B.linalg.vector_norm(sphere_point, dim=-1, keepdim=True)
            x, y, z, w = (sphere_point[..., i].unsqueeze(-1) for i in range(4))
            xx = x ** 2
            yy = y ** 2
            zz = z ** 2
            xy = x * y
            xz = x * z
            xw = x * w
            yz = y * z
            yw = y * w
            zw = z * w
            del sphere_point, x, y, z, w
            r1 = B.hstack((1-2*(yy+zz), 2*(xy-zw), 2*(xz+yw))).unsqueeze(-1)
            r2 = B.hstack((2*(xy+zw), 1-2*(xx+zz), 2*(yz-xw))).unsqueeze(-1)
            r3 = B.hstack((2*(xz-yw), 2*(yz+xw), 1-2*(xx+yy))).unsqueeze(-1)
            del xx, yy, zz, xy, xz, xw, yz, yw, zw
            q = B.cat((r1, r2, r3), -1)
            return q
        else:
            h = B.random.randn(key, dtype_double(key), number, self.n, self.n)
            q, r = B.linalg.qr(h)
            r_diag_sign = B.sign(B.diagonal(r, dim1=-2, dim2=-1))
            q *= r_diag_sign[:, None]
            q_det_sign = B.sign(B.det(q))
            q[:, :, 0] *= q_det_sign[:, None]
            return q