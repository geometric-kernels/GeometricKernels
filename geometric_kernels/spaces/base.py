"""
Abstract base interface for spaces.
"""
import abc
from typing import Callable

from geometric_kernels.types import TensorLike


class Space(abc.ABC):
    """
    Object representing a space on which a kernel can be defined.
    """

    @abc.abstractproperty
    def dimension(self) -> int:
        """Dimension in which the space is embedded"""
        raise NotImplementedError
    

class SpaceWithEigenDecomposition(Space):
    """
    A Space for which we can obtain the eigenvalues and eigenfunctions of
    the Laplace-Beltrami operator.

    Examples includes `Graph`s, `Manifold`s and `Mesh`es.
    """

    @abc.abstractmethod
    def get_eigenfunctions(self, num: int) -> Callable[[TensorLike], TensorLike]:
        """
        First `num` eigenfunctions of the Laplace-Beltrami operator
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_eigenvalues(self, num: int) -> TensorLike:
        """
        First `num` eigenvalues of the Laplace-Beltrami operator

        :return: [num, 1] array containing the eigenvalues
        """
        raise NotImplementedError
