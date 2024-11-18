"""
Abstract base classes for all spaces (input domains) in the library.
"""

import abc

import lab as B
from beartype.typing import List

from geometric_kernels.spaces.eigenfunctions import Eigenfunctions


class Space(abc.ABC):
    """
    A space (input domain) on which a geometric kernel can be defined.
    """

    @abc.abstractproperty
    def dimension(self) -> int:
        """
        Geometric dimension of the space.

        Examples:

        * :class:`~.spaces.Graph`: 0-dimensional.
        * :class:`~.spaces.Circle`: 1-dimensional.
        * :class:`~.spaces.Hypersphere`: d-dimensional, with d >= 2.
        * :class:`~.spaces.Hyperbolic`: d-dimensional, with d >= 2.
        """
        raise NotImplementedError

    @abc.abstractproperty
    def element_shape(self) -> List[int]:
        """
        Shape of an element.

        Examples:
        * :class:`~.spaces.Hypersphere`: [D + 1, ]
        * :class:`~.spaces.Mesh`: [1, ]
        * :class:`~.spaces.CompactMatrixLieGroup`: [n, n]
        """
        raise NotImplementedError

    @abc.abstractproperty
    def element_dtype(self) -> B.DType:
        """
        Abstract DType of an element.

        Examples:
        * :class:`~.spaces.Hypersphere`: B.Float
        * :class:`~.spaces.Mesh`: B.Int
        * :class:`~.spaces.SpecialUnitary`: B.Complex
        """
        raise NotImplementedError


class DiscreteSpectrumSpace(Space):
    r"""
    A Space with discrete spectrum (of the Laplacian operator).

    This includes, for instance, compact Riemannian manifolds, graphs & meshes.

    Subclasses implement routines for computing the eigenvalues and
    eigenfunctions of the Laplacian operator, or certain combinations thereof.
    Since there is often an infinite or a prohibitively large number of those,
    they only compute a finite subset, consisting of the ones that are most
    important for approximating Matérn kernel best.

    .. note::
        See a brief introduction into the theory behind the geometric
        kernels on discrete spectrum spaces on the documentation pages devoted
        to :doc:`compact Riemannian manifolds </theory/compact>` (also
        :doc:`this </theory/addition_theorem>`), :doc:`graphs
        </theory/graphs>` and :doc:`meshes </theory/meshes>`.

    .. note::
        Typically used with :class:`~.kernels.MaternKarhunenLoeveKernel`.

    """

    @abc.abstractproperty
    def dimension(self) -> int:
        """
        Geometric dimension of the space.

        Examples:

        * :class:`~.spaces.Graph`: 0-dimensional.
        * :class:`~.spaces.Circle`: 1-dimensional.
        * :class:`~.spaces.Hypersphere`: d-dimensional, with d >= 2.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        Returns the :class:`~.Eigenfunctions` object with `num` levels.

        :param num:
            Number of levels.

        .. note::
            The notion of *levels* is discussed in the documentation of the
            :class:`~.kernels.MaternKarhunenLoeveKernel`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_eigenvalues(self, num: int) -> B.Numeric:
        """
        Eigenvalues of the Laplacian corresponding to the first `num` levels.

        :param num:
            Number of levels.
        :return:
            (num, 1)-shaped array containing the eigenvalues.

        .. note::
            The notion of *levels* is discussed in the documentation of the
            :class:`~.kernels.MaternKarhunenLoeveKernel`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_repeated_eigenvalues(self, num: int) -> B.Numeric:
        """
        Eigenvalues of the Laplacian corresponding to the first `num` levels,
        repeated according to their multiplicity within levels.

        :param num:
            Number of levels.

        :return:
            (J, 1)-shaped array containing the repeated eigenvalues, J is
            the resulting number of the repeated eigenvalues.

        .. note::
            The notion of *levels* is discussed in the documentation of the
            :class:`~.kernels.MaternKarhunenLoeveKernel`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def random(self, key: B.RandomState, number: int) -> B.Numeric:
        """
        Sample uniformly random points in the space.

        :param key:
            Either `np.random.RandomState`, `tf.random.Generator`,
            `torch.Generator` or `jax.tensor` (representing random state).
        :param number:
            Number of samples to draw.

        :return:
            An array of `number` uniformly random samples on the space.
        """
        raise NotImplementedError


class NoncompactSymmetricSpace(Space):
    """
    Non-compact symmetric space.

    This includes, for instance, hyperbolic spaces and manifolds of symmetric
    positive definite matrices (endowed with the affine-invariant metric).

    .. note::
        See a brief introduction into the theory behind the geometric
        kernels on non-compact symmetric spaces on the
        :doc:`respective documentation page </theory/symmetric>`.

    .. note::
        Typically used with :class:`~.kernels.MaternFeatureMapKernel` that
        builds on a space-specific feature map like the
        :class:`~.feature_maps.RejectionSamplingFeatureMapHyperbolic` and the
        :class:`~.feature_maps.RejectionSamplingFeatureMapSPD`, or, in the
        absence of a space-specific feature map, on the general (typically less
        effective) map :class:`~.feature_maps.RandomPhaseFeatureMapNoncompact`.

    .. note:: .. _quotient note:

        Mathematically, any non-compact symmetric space can be represented as
        a quotient $G/H$ of a Lie group of symmetries $G$ and its compact
        isotropy subgroup $H$. We sometimes refer to these $G$ and $H$ in
        the documentation. See mathematical details in :cite:t:`azangulov2024b`.
    """

    @abc.abstractproperty
    def dimension(self) -> int:
        """
        Geometric dimension of the space.

        Examples:

        * :class:`~.spaces.Hyperbolic`: d-dimensional, with d >= 2.
        * :class:`~.spaces.SymmetricPositiveDefiniteMatrices`: $n(n+1)/2$-dimensional,
            with n >= 2.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def inv_harish_chandra(self, lam: B.Numeric) -> B.Numeric:
        r"""
        Implements $c^{-1}(\lambda)$, where $c$ is the Harish-Chandra's $c$
        function.

        This is one of the computational primitives required to (approximately)
        compute the :class:`~.feature_maps.RandomPhaseFeatureMapNoncompact`
        feature map and :class:`~.kernels.MaternFeatureMapKernel` on top of it.

        :param lam:
            A batch of frequencies, vectors of dimension equal to the rank of
            symmetric space.

        :return:
            $c^{-1}(\lambda)$ evaluated at every $\lambda$ in the batch `lam`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def power_function(self, lam: B.Numeric, g: B.Numeric, h: B.Numeric) -> B.Numeric:
        r"""
        Implements the *power function* $p^{\lambda}(g, h)$, the integrand
        appearing in the definition of the zonal spherical function

        .. math:: \pi^{\lambda}(g) = \int_{H} \underbrace{p^{\lambda}(g, h)}_{= e^{(i \lambda + \rho) a(h \cdot g)}} d h,

        where $\lambda \in i \cdot \mathbb{R}^r$, with $r$ denoting the rank of
        the symmetric space and $i$ the imaginary unit, is a sort of frequency,
        $g$ is an element of the group of symmetries $G$, $h$ is an element
        of its isotropy subgroup $H$ ($G$ and $H$ are defined :ref:`here
        <quotient note>`), $\rho \in \mathbb{R}^r$ is as in :meth:`rho`, and
        the function $a$ is a certain space-dependent algebraic operation.

        This is one of the computational primitives required to (approximately)
        compute the :class:`~.feature_maps.RandomPhaseFeatureMapNoncompact`
        feature map and :class:`~.kernels.MaternFeatureMapKernel` on top of it.

        :param lam:
            A batch of L vectors of dimension `rank`, the rank of the
            symmetric space, representing the "sort of frequencies".

            Typically of shape [1, L, rank].
        :param g:
            A batch of N elements of the space (these can always be thought of
            as elements of the group of symmetries $G$ since the symmetric
            space $G/H$ can be trivially embedded into the group $G$).

            Typically of shape [N, 1, <axes>], where <axes> is the shape of
            the elements of the space.
        :param h:
            A batch of L elements of the isotropy subgroup $H$.

            Typically of shape [1, L, <axes_p>], where <axes_p> is the shape of
            arrays representing the elements of the isotropy subgroup $H$.

        :return:
            An array of shape [N, L] with complex number entries, representing
            the value of the values of $p^{\lambda_l}(g_n, h_l)$ for all
            $1 \leq n \leq N$ and $1 \leq l \leq L$.

        .. note::
            Actually, $a$ may be a more appropriate primitive than the power
            function $p^{\lambda}$: everything but $a$ in the definition of
            the latter is either standard or available as other primitives.
            Us using $p^{\lambda}$ as a primitive is quite arbitrary.
        """
        raise NotImplementedError

    @abc.abstractproperty
    def rho(self):
        r"""
        `rho` vector of dimension equal to the rank of the symmetric space.

        Algebraically, weighted sum of *roots*, depends only on the space.

        This is one of the computational primitives required to (approximately)
        compute the :class:`~.feature_maps.RandomPhaseFeatureMapNoncompact`
        feature map and :class:`~.kernels.MaternFeatureMapKernel` on top of it.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def random_phases(self, key: B.RandomState, num: int) -> B.Numeric:
        r"""
        Sample uniformly random points on the isotropy subgroup $H$ (defined
        :ref:`here <quotient note>`).

        This is one of the computational primitives required to (approximately)
        compute the :class:`~.feature_maps.RandomPhaseFeatureMapNoncompact`
        feature map and :class:`~.kernels.MaternFeatureMapKernel` on top of it.

        :param key:
            Either `np.random.RandomState`, `tf.random.Generator`,
            `torch.Generator` or `jax.tensor` (representing random state).
        :param num:
            Number of samples to draw.

        :return:
            An array of `num` uniformly random samples in the isotropy
            subgroup $H$.

        .. warning::
            This does not sample random points on the space itself. Since the
            space itself is non-compact, uniform sampling on it is in principle
            impossible. However, the isotropy subgroup $H$ is always
            compact and thus allows uniform sampling needed to approximate the
            zonal spherical functions $\pi^{\lambda}(\cdot)$ via Monte Carlo.
        """

    @abc.abstractproperty
    def num_axes(self):
        """
        Number of axes in an array representing a point in the space.

        Usually 1 for vectors or 2 for matrices.
        """
