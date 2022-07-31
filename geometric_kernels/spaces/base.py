"""
Abstract base interface for spaces.
"""
import abc

import lab as B

from geometric_kernels.spaces.eigenfunctions import (
    Eigenfunctions,
    EigenfunctionWithAdditionTheorem,
)
from geometric_kernels.lab_extras import take_along_axis


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

    def num_eigenfunctions(self) -> int:
        """Number of eigenvectors, M"""
        return B.shape(self.eigenvectors)[-1]
