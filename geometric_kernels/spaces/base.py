import abc

import numpy as np


class Space(abc.ABC):
    """
    Object representing a space on which a kernel can be defined. Needs to provide
    a method to obtain the eigendecomposition of the space w.r.t the Laplace-Beltrami
    operator.

    Examples are `Graph`, `Manifold` and `Mesh`.
    """

    @abc.abstractproperty
    def dim(self) -> int:
        """Dimension in which the space is embedded"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_eigenfunctions(self, num: int):
        """
        First `num` eigenfunctions of the Laplace-Beltrami operator
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_eigenvalues(self, num: int) -> np.ndarray:
        """
        First `num` eigenvalues of the Laplace-Beltrami operator

        :return: [num, 1] array containing the eigenvalues
        """
        raise NotImplementedError
