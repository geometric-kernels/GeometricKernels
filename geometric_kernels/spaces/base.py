"""
Abstract base interface for GeometricKernels spaces.
"""

import abc

import lab as B
from beartype.typing import List

from geometric_kernels.lab_extras import take_along_axis
from geometric_kernels.spaces.eigenfunctions import Eigenfunctions


class Space(abc.ABC):
    """
    Object representing a space on which a kernel can be defined.
    """

    @abc.abstractproperty
    def dimension(self) -> int:
        """
        Dimension of the manifold

        Examples:

        * circle: 1 dimensional
        * sphere: 2 dimensional
        * torus: 2 dimensional
        """
        raise NotImplementedError

    @abc.abstractproperty
    def element_shape(self) -> List[int]:
        """
        Shape of an element.

        Examples:
        * hypersphere: [D + 1, ]
        * mesh: [1, ]
        * matrix Lie group: [n, n]
        """
        raise NotImplementedError


class DiscreteSpectrumSpace(Space):
    r"""
    A Space for which we can obtain the eigenvalues and eigenfunctions of
    the Laplace-Beltrami operator.

    Examples includes `Graph`\s, `Manifold`\s and `Mesh`\es.
    """

    @abc.abstractmethod
    def get_eigenfunctions(self, num: int) -> Eigenfunctions:
        """
        Eigenfunctions of the Laplace-Beltrami operator corresponding
        to the first `num` levels.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_eigenvalues(self, num: int) -> B.Numeric:
        """
        Eigenvalues of the Laplace-Beltrami operator corresponding
        to the first `num` levels.

        :return: [num, 1] array containing the eigenvalues.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_repeated_eigenvalues(self, num: int) -> B.Numeric:
        """Eigenvalues of the Laplace-Beltrami operator that
        correspond to the first `num` levels, repeated according to
        the number of eigenfunctions within a level.

        :return: [M, 1] array containing the eigenvalues

        """
        raise NotImplementedError

    @abc.abstractmethod
    def random(self, key, number: int) -> B.Numeric:
        """
        Return randomly sampled points in the space
        """
        raise NotImplementedError


class NoncompactSymmetricSpace(Space):
    """
    Non-compact symmetric space.

    Examples include the `Hyperbolic` space and `SPD` space.
    """

    @abc.abstractmethod
    def inv_harish_chandra(self, X: B.Numeric) -> B.Numeric:
        """
        (Multiplicative) inverse of the Harish-Chandra function.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def power_function(self, lam, g, h) -> B.Numeric:
        r"""
        Power function :math:`p^{\lambda)(g, h) = \exp(i \lambda + \rho) a(h \cdot g)`.

        Zonal spherical functions are defined as :math:`\pi^{\lambda}(g) = \int_{H} p^{\lambda}(g, h) d\mu_H(h)`
        """
        raise NotImplementedError

    @abc.abstractproperty
    def rho(self):
        r"""
        `rho` vector.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def random_phases(self, key, num) -> B.Numeric:
        r"""
        Random samples from Haar measure on the isotropy group of the symmetric space.
        """

    @abc.abstractproperty
    def num_axes(self):
        """
        Number of axes in an array representing a point in the space.
        Ususally 1 for vectors and 2 for matrices.
        """


class ConvertEigenvectorsToEigenfunctions(Eigenfunctions):
    """
    Converts the array of eigenvectors to callable objects,
    where inputs are given by the indices. Based on
    from geometric_kernels.spaces.mesh import ConvertEigenvectorsToEigenfunctions.
    """

    def __init__(self, eigenvectors: B.Numeric):
        """
        :param eigenvectors: [Nv, M]
        """
        self.eigenvectors = eigenvectors

    def __call__(self, X: B.Numeric, **parameters) -> B.Numeric:
        """
        Selects `N` locations from the `M` eigenvectors.

        :param X: indices [N, 1]
        :param parameters: unused
        :return: [N, M]
        """
        indices = B.cast(B.dtype_int(X), X)
        Phi = take_along_axis(self.eigenvectors, indices, axis=0)
        return Phi

    @property
    def num_eigenfunctions(self) -> int:
        """Number of eigenvectors, M"""
        return B.shape(self.eigenvectors)[-1]

    @property
    def num_levels(self) -> int:
        """Number of levels, L"""
        return self.num_eigenfunctions

    @property
    def num_eigenfunctions_per_level(self) -> B.Numeric:
        return [1] * self.num_levels
