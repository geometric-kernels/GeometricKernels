"""
This module provides the :class:`Grassmannian` space and the representation of
its spectrum, the :class:`GrassmannianEigenfunctions` class.
"""

import json

import lab as B
import numpy as np
import sympy
from beartype.typing import List, Optional, Tuple
from lab import einsum

from geometric_kernels.lab_extras import dtype_double, from_numpy, qr
from geometric_kernels.spaces import DiscreteSpectrumSpace
from geometric_kernels.spaces.eigenfunctions import EigenfunctionsWithAdditionTheorem
from geometric_kernels.spaces.so import SOEigenfunctions, SpecialOrthogonal
from geometric_kernels.utils.utils import get_resource_file_path


class GrassmannianStabilizer:
    """
    Helper class for sampling from Grassmannian stabilizer that is represented as S(O(n) x O(m))
    by (n+m) x (n+m) block-diagonal matrices.
    """

    def __init__(self, n: int, m: int):
        self.n, self.m = n, m
        self.so_n = SpecialOrthogonal(n)
        self.so_m = SpecialOrthogonal(m)
        self.dim = self.so_n.dim + self.so_m.dim

    def random(self, key, number):
        """Randomly samples `number` matrices of size (n+m) x (n+m).

        Each sample has a form of `[[H_n, 0], [0, H_m]]`. The upper left block
        is uniformly sampled over O(n), and the lower right block is
        uniformly sampled over O(m), and the signs of the blocks are adjusted.
        """
        key, sign = B.randint(key, dtype_double(key), number, lower=0, upper=2)
        sign = 2 * sign - 1  # convert to -1, 1
        key, h_u = self.so_n.random(key, number)  # [number, n, n]
        key, h_d = self.so_m.random(key, number)  # [number, m, m]
        h_u[:, :, -1] *= sign[
            :, None
        ]  # Ensure the last column of h_u has the same sign as the block
        h_d[:, :, -1] *= sign[
            :, None
        ]  # Ensure the last column of h_d has the same sign as the block

        zeros = B.zeros(B.dtype(h_u), number, self.n, self.m)  # [number, n, m]
        zeros_t = B.transpose(zeros)

        # [number, n + m, n], [number, n + m, m]
        left_block = B.concat(h_u, zeros_t, axis=-2)
        right_block = B.concat(zeros, h_d, axis=-2)
        res = B.concat(left_block, right_block, axis=-1)  # [number, n + m, n + m]
        return key, res


class GrassmannianZonalSphericalFunction:
    """
    The class that represents a zonal spherical function on the Grassmannian

    These are polynomials whose coefficients are precomputed and stored in a
    file. By default, there are 20 precomputed characters for n from 3 to 10.
    If you want more, use the `compute_grassmannian_zsf.py` script.

    :param n:
        The order n of the Gr(n, m).
    :param m:
        The order m of the Gr(n, m).
    :param signature:
        The signature that determines a particular character (and an
        irreducible unitary representation along with it).
    """

    def __init__(self, n: int, m: int, signature: Tuple[int, ...]):
        self.signature = signature
        self.n = n
        self.m = m
        self.coeffs, self.monoms = self._load()
        self.normalization = np.sum(
            self.coeffs
        )  # Normalization factor for the character

    def _load(self):
        group_name = "Gr({},{})".format(self.n, self.m)
        with get_resource_file_path("precomputed_grassmanian_zsf.json") as file_path:
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
        return char_val / self.normalization


class GrassmannianEigenfunctions(EigenfunctionsWithAdditionTheorem):
    def __init__(self, space, num_levels, compute_zsf=True):
        super().__init__()
        self.space = space
        self.n = space.n
        self.m = space.m
        self._num_levels = num_levels
        self.rank = min(self.m, self.n - self.m)
        self.G_rank = space.G.rank
        self.G_eigenfunctions = SOEigenfunctions(
            self.n, num_levels=0, compute_characters=False
        )

        self._signatures = self._generate_signatures(num_levels)
        self._eigenvalues = np.array(
            [self._compute_eigenvalue(sgn) for sgn in self._signatures]
        )
        self.repr_dim = [
            (1 + (signature[-1] != 0))
            * self.G_eigenfunctions._compute_dimension(signature)
            for signature in self._signatures
        ]
        if compute_zsf:
            self._zsf = [
                self._compute_zsf(self.n, self.m, sgn) for sgn in self._signatures
            ]

        self._num_eigenfunctions = None

    def _generate_signatures(self, num_levels: int) -> List[Tuple[int, ...]]:
        eigen_map = {}
        rank = self.rank
        if rank <= 1:
            SIGNATURE_SUM = 100
        else:
            SIGNATURE_SUM = 20
        if num_levels <= 0:
            return []

        if rank <= 0:
            raise ValueError("m must be less than n")
        # Degree 0: The empty partition
        sgn_trivial = tuple([0] * self.G_rank)
        eigen_map[sgn_trivial] = 0.0

        # Iterate through degrees (sum of parts of the partition)
        for degree in range(1, SIGNATURE_SUM):
            for p_dict in sympy.utilities.iterables.partitions(degree):
                partition_list = []
                for val, count in sorted(p_dict.items(), reverse=True):
                    partition_list.extend([val] * count)
                kappa = list(partition_list)

                if (
                    len(kappa) <= rank
                ):  # Filter by max allowed length for this Grassmannian
                    kappa = kappa + [0] * (
                        self.G_rank - len(kappa)
                    )  # Pad with zeros to match rank
                    kappa_even = all(x % 2 == 0 for x in kappa)
                    sgn = tuple(kappa)
                    if sgn not in eigen_map and kappa_even:
                        eigen_map[sgn] = self._compute_eigenvalue(sgn)

        # Convert to list of (eigenvalue, kappa) for sorting
        # Sort by eigenvalue, then by partition degree, then by partition tuple (lexicographically)
        # for stable and canonical tie-breaking.
        sorted_eigenpairs = sorted(
            [(eig, sgn) for sgn, eig in eigen_map.items()],
            key=lambda x: (x[0], sum(x[1]), x[1]),
        )
        signatures = [sgn for _, sgn in sorted_eigenpairs]
        return signatures[:num_levels]

    def _compute_eigenvalue(self, signature):
        """
        Computes the eigenvalue of the Laplace-Beltrami operator.
        """
        if signature[0] == 0:
            return 0.0

        eigenvalue = 0.0
        for j_idx, sgn_j in enumerate(signature):
            if signature[j_idx] == 0:
                break
            j = j_idx + 1
            eigenvalue += sgn_j * (sgn_j + self.n - 2 * j)
        return eigenvalue

    def _compute_zsf(
        self, n: int, m: int, signature: Tuple[int, ...]
    ) -> GrassmannianZonalSphericalFunction:
        return GrassmannianZonalSphericalFunction(n, m, signature)

    def _difference(self, X: B.Numeric, X2: B.Numeric) -> B.Numeric:
        """
        Pairwise differences between points of the homogeneous space M
        embedded into G.

        :param X:
            [N1, ...] an array of points in `M`.
        :param X2:
            [N2, ...] an array of points in `M`.

        :return:
            [N1, N2, ...] an array of points in `G`.
        """
        g = self.space.embed_manifold(X)
        g2 = self.space.embed_manifold(X2)
        diff = self.G_eigenfunctions._difference(g, g2, inverse_X=True)
        diff = self.space.project_to_manifold(diff)
        return diff

    def _torus_representative(self, X: B.Numeric, **kwargs) -> B.Numeric:
        """
        Computes the torus representative of the difference between two matrices.
        For the Grassmannian, this is just the difference itself.
        """
        # X has shape [..., n, m]
        X_ = X[..., : self.m, :]  # [..., m, m]
        X_T_X = einsum("...ji,...jk->...ik", X_, X_)
        eigvals = B.eig(X_T_X, False)
        sorted_ind = B.argsort(B.real(eigvals), axis=-1)
        eigvals = take_along_axis(eigvals, sorted_ind, -1)[
            ..., : self.rank
        ]  # Sort eigenvalues
        return eigvals

    def _addition_theorem(
        self, X: B.Numeric, X2: Optional[B.Numeric] = None, **kwargs
    ) -> B.Numeric:
        r"""For each level (that corresponds to a unitary irreducible
        representation of the group), computes the sum of outer products of
        Laplace-Beltrami eigenfunctions that correspond to this level
        (representation). Uses the fact that such sums are equal to the
        character of the representation multiplied by the dimension of that
        representation. See :cite:t:`azangulov2024a` for mathematical details.

        :param X:
            An [N, n, n]-shaped array, a batch of N matrices of size nxn.
        :param X2:
            An [N2, n, n]-shaped array, a batch of N2 matrices of size nxn.

            Defaults to None, in which case X is used for X2.
        :param ``**kwargs``:
            Any additional parameters.

        :return:
            An array of shape [N, N2, L].
        """
        if X2 is None:
            X2 = X
        diff = self._difference(X, X2)
        torus_repr_diff = self._torus_representative(diff)
        values = [
            repr_dim * zsf(torus_repr_diff)[..., None]  # [N, N2, 1]
            for zsf, repr_dim in zip(self._zsf, self.repr_dim)
        ]
        return B.concat(*values, axis=-1)  # [N, N2, L]

    def _addition_theorem_diag(self, X: B.Numeric, **kwargs) -> B.Numeric:
        """
        A more efficient way of computing the diagonals of the matrices
        `self._addition_theorem(X, X)[:, :, l]` for all l from 0 to L-1.

        :param X:
            As in :meth:`_addition_theorem`.
        :param ``**kwargs``:
            As in :meth:`_addition_theorem`.

        :return:
            An array of shape [N, L].
        """
        ones = B.ones(B.dtype(X), *X.shape[:-2], 1)
        values = [
            repr_dim * ones  # [N, 1], because chi(X*inv(X))=repr_dim
            for repr_dim in self.repr_dim
        ]
        return B.concat(*values, axis=1)  # [N, L]

    @property
    def num_eigenfunctions_per_level(self) -> List[int]:
        """
        The number of eigenfunctions per level.

        :return:
            List of ones of length num_levels.
        """
        return self.repr_dim

    @property
    def num_eigenfunctions(self) -> int:
        if self._num_eigenfunctions is None:
            self._num_eigenfunctions = sum(self.num_eigenfunctions_per_level)
        return self._num_eigenfunctions

    @property
    def num_levels(self) -> int:
        return self._num_levels


class Grassmannian(DiscreteSpectrumSpace):
    r"""
    The GeometricKernels space representing the Grassmannian manifold Gr(n, m)
    as the homogeneous space O(n) / (O(m) x O(n-m)) which also happens
    to be a symmetric space.

    The elements of this space are represented as n x m matrices
    with orthogonal columns, just like the elements of the :class:`Stiefel`
    space. However, for this space, this representation is not unique: two such
    matrices can represent the same element of the Grassmannian manifold.

    .. note::
        A tutorial on how to use this space is available in the
        :doc:`Grassmannian.ipynb </examples/Grassmannian>` notebook.

    .. admonition:: Citation

        If you use this GeometricKernels space in your research, please consider
        citing :cite:t:`azangulov2022`.
    """

    def __init__(self, n: int, m: int):
        """
        :param n:
            The number of rows.
        :param m:
            The number of columns.
        :param key:
            Random state used to sample from the stabilizer.
        :param average_order:
            The number of random samples from the stabilizer.

        :return:
            A tuple (new random state, a realization of Gr(m, n)).
        """
        assert n > m, "n should be greater than m"
        assert (m > 1) and (
            m < n - 1
        ), "Isomorphic to hypersphere, use Hypersphere class instead"

        super().__init__()
        self.H = GrassmannianStabilizer(m, n - m)
        self.G = SpecialOrthogonal(n)
        self.dim_H = self.H.dim
        self.n = n
        self.m = m

    def project_to_manifold(self, g):
        """
        Take first m columns of an orthogonal matrix.

        :param g:
            [..., n, n] array of points in SO(n).

        :return:
            [..., n, m] array of points in V(n, m).
        """

        return g[..., : self.m]

    def embed_manifold(self, x):
        """
        Complete [n, m] matrix with orthogonal columns to an orthogonal
        [n, n] one using QR algorithm.

        :param x:
            [..., n, m] array of points in V(m, n).

        :return g:
            [..., n, n] array of points in SO(n).
        """
        if x.shape[-1] == self.n:
            # If x is already in SO(n), just return it
            return x

        p = B.matmul(x, B.transpose(x, [0, 2, 1]))  # Shape: (b, n, n)
        r = B.randn(B.dtype(x), *x.shape[:-1], self.n - self.m)  # Shape: (b, n, n - m)

        r_orth = r - B.matmul(p, r)  # (b, n, n - m)

        q, _ = qr(r_orth)  # (b, n, n - m)

        g = B.concat(x, q, axis=2)  # (b, n, n)
        det = B.sign(B.det(g))
        g[:, :, -1] *= det[:, None]
        return g

    def embed_stabilizer(self, h):
        """
        Embed SO(m) x SO(n-m) matrix into SO(n),
        In case of the Grassmannian, this is an identity mapping.

        :param h:
            [..., n, n] array of points in SO(m) x SO(n-m).

        :return:
            [..., n, n] array of points in SO(n).
        """
        return h

    def get_eigenfunctions(self, num: int) -> GrassmannianEigenfunctions:
        eigenfunctions = GrassmannianEigenfunctions(self, num)
        return eigenfunctions

    def get_repeated_eigenvalues(self, num: int) -> B.Numeric:
        return self.get_eigenvalues(num)

    def get_eigenvalues(self, num: int) -> B.Numeric:
        eigenfunctions = GrassmannianEigenfunctions(self, num)
        eigenvalues = np.array(eigenfunctions._eigenvalues)
        return B.reshape(eigenvalues, -1, 1)  # [num, 1]

    @property
    def element_shape(self):
        """
        :return:
            [n, m].
        """
        return [self.n, self.m]

    @property
    def element_dtype(self):
        """
        Return the data type of the elements of the Grassmannian.
        """
        return B.Float

    @property
    def dimension(self):
        """
        Return the dimension of the Grassmannian.
        """
        return self.m * (self.n - self.m)

    def random(self, key, number: int, project=False):
        """
        Samples random points from the uniform distribution on M.

        :param key:
            A random state.
        :param number:
            A number of random to generate.

        :return:
            [number, ...] an array of randomly generated points.
        """
        key, raw_samples = self.G.random(key, number)
        if project:
            return key, self.project_to_manifold(raw_samples)
        else:
            return key, raw_samples
