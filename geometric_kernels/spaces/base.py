"""
Abstract base interface for spaces.
"""
import abc

import lab as B

from geometric_kernels.eigenfunctions import (
    Eigenfunctions,
    EigenfunctionWithAdditionTheorem,
)


class Space(abc.ABC):
    """
    Object representing a space on which a kernel can be defined.
    """

    @abc.abstractproperty
    def dimension(self) -> int:
        """
        Dimension of the manifold
        Examples:
         - circle: 1
         - sphere: 2
         - torus: 2
        """
        raise NotImplementedError


class DiscreteSpectrumSpace(Space):
    """
    A Space for which we can obtain the eigenvalues and eigenfunctions of
    the Laplace-Beltrami operator.

    Examples includes `Graph`s, `Manifold`s and `Mesh`es.
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


class DiscreteSpectrumSpaceWithAdditionTheorem(DiscreteSpectrumSpace):
    """In addition to the DiscreteSpecturmSpace properties, there
    exists an addition theorem for the eigenfunctions of this space
    such that
        \sum_{i \in level} eigenfunction_i(x) eigenfunction_i(y) = addition_function_i(x,y)
    """

    @abc.abstractmethod
    def get_eigenfunctions_from_levels(
        self, num: int
    ) -> EigenfunctionWithAdditionTheorem:
        """
        First `num` levels of eigenfunctions of the Laplace-Beltrami operator
        """
        raise NotImplementedError
