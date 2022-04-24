"""
Abstract base interface for spaces.
"""
import abc

import lab as B

from geometric_kernels.eigenfunctions import Eigenfunctions


class Space(abc.ABC):
    """
    Object representing a space on which a kernel can be defined.
    """

    @abc.abstractproperty
    def dimension(self) -> int:
        """
        Dimension of the manifold

        Examples:

        * circle: 1
        * sphere: 2
        * torus: 2
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
        First `num` eigenfunctions of the Laplace-Beltrami operator
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_eigenvalues(self, num: int) -> B.Numeric:
        """
        First `num` eigenvalues of the Laplace-Beltrami operator

        :return: [num, 1] array containing the eigenvalues
        """
        raise NotImplementedError
