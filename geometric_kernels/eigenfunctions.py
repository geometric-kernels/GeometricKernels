from typing import Optional

import abc
import tensorflow as tf
import numpy as np

from geometric_kernels.types import TensorLike


class Eigenfunctions(abc.ABC):
    r"""
    Represents a set of eigenfunctions of an operator. Referred to as :math:`Phi = [\phi_i]_{i=0}^{M-1}`.
    """
    def weighted_outerproduct(self, weights: TensorLike, X: TensorLike, X2: Optional[TensorLike] = None) -> TensorLike:
        r"""
        Computes :math:`\sum_{i=0}^{M-1} w_i \phi_i(x1) \phi_i(x2)`.

        :param weights: [M, 1]
        :param X: Inputs where to evaluate the eigenfunctions, shape = [N, D].
        :param X2: Inputs where to evaluate the eigenfunctions, shape = [N2, D].
            Default to None, in which X is used for X2.
        :return: shape [N, N2]
        """
        Phi_X = self.__call__(X)  # [N, M]
        if X2 is None:
            Phi_X2 = Phi_X
        else:
            Phi_X2 = self.__call__(X2)  # [N2, M]

        return tf.matmul(Phi_X, tf.transpose(weights) * Phi_X2, transpose_b=True)  # [N, N2]

    def weighted_outerproduct_diag(self, weights: TensorLike, X: TensorLike) -> TensorLike:
        r"""
        Computes :math:`\sum_{i=0}^{M-1} w_i \phi_i(x) \phi_i(x)`. Corresponds to the
        diagonal elements of `weighted_outproduct` but they can be calculated more
        efficiently.

        :param weights: [M, 1]
        :param X: Inputs where to evaluate the eigenfunctions, shape = [N, D].
        :return: shape [N,]
        """
        Phi_X = self.__call__(X)  # [N, L]
        return tf.reduce_sum(Phi_X ** 2 * tf.transpose(weights), axis=1)  # [N,]

    @abc.abstractmethod
    def __call__(self, x: TensorLike) -> TensorLike:
        raise NotImplementedError
    
    @abc.abstractproperty
    def num_eigenfunctions(self) -> int:
        """Number of eigenfunctions, M"""
        raise NotImplementedError
    

class EigenfunctionWithAdditionTheorem(Eigenfunctions):
    """
    Eigenfunctions for which the sum over a level has a simpler expression.
    In the case the weights over a level in the `weighted_outproduct` our identical
    we can make use of this expression to simplify computations.

    We assume there are `L` levels. The sum of the number of eigenfunctions per level should
    equal the total amount of eigenfunctions.
    """
    def __post_init__(self) -> None:
        """Checks"""
        num_eigenfunctions_manual = np.sum(self.num_eigenfunctions_per_level)
        assert num_eigenfunctions_manual == self.num_eigenfunctions
        assert len(self.num_eigenfunctions_per_level) == self.num_levels

    def weighted_outer_product(self, weights: TensorLike, X: TensorLike, X2: Optional[TensorLike] = None) -> TensorLike:
        r"""
        Computes :math:`\sum w_i \phi_i(x1) \phi_i(x2)`.

        :param weights: [L, 1]
            .. note:
                The length of `weights` is equal to the number of levels.
                This is **not** the same as the number of eigenfunctions.

        :param X: Inputs where to evaluate the eigenfunctions, shape = [N, D].
        :param X2: Inputs where to evaluate the eigenfunctions, shape = [N2, D].
            Default to None, in which X is used for X2.
        :return: shape [N, N2]
        """
        if X2 is None:
            sum_phi_phi_for_level = self.addition_theorem(X, X)  # [N, N, L]
        else:
            sum_phi_phi_for_level = self.addition_theorem(X, X2)  # [N, N2, L]

        w = tf.transpose(weights)[None]  # [1, 1, L]
        return tf.reduce_sum(w * sum_phi_phi_for_level, axis=2)  # [N, N2]

    def weighted_outerproduct_diag(self, weights: TensorLike, X: TensorLike) -> TensorLike:
        r"""
        Computes :math:`\sum_{i=0}^{M-1} w_i \phi_i(x) \phi_i(x)`. Corresponds to the
        diagonal elements of `weighted_outproduct` but they can be calculated more
        efficiently.

        Makes use of the fact that eigenfunctions within a level can be summed
        in a computationally more efficient matter.

        .. note:
            Only works if the weights within a level is equal.

        :param weights: [M, 1]
        :param X: Inputs where to evaluate the eigenfunctions, shape = [N, D].
        :return: shape [N,]
        """
        weights_per_level = []
        # select the weight for each level
        # assumes the weights in `weights` within a level is identical
        # TODO(VD) write check for this.
        i = 0
        for num in self.num_eigenfunctions_per_level:
            weights_per_level.append(weights[i])
            i += num

        weights = tf.reshape(tf.convert_to_tensor(weights_per_level), (-1, 1))
        addition_theorem_X = self.addition_theorem(X)  # [N, L]
        return tf.reduce_sum(addition_theorem_X * tf.transpose(weights), axis=1)  # [N,]
    
    @abc.abstractmethod
    def addition_theorem(self, X: TensorLike, X2: Optional[TensorLike] = None) -> TensorLike:
        """
        Returns the sum of eigenfunctions on a level for which we have a simplified expression

        :param X: [N, D]
        :param X2: [N2, D], default to None
        :return: Evaluate the sum of eigenfunctions on each level. Returns
            a value for each level [N, N2, L] if `X2` not `None`, else [N, L].
        """
        raise NotImplementedError

    @abc.abstractproperty
    def num_levels(self) -> int:
        """Number of levels, L"""
        raise NotImplementedError

    @abc.abstractproperty
    def num_eigenfunctions_per_level(self) -> np.ndarray:
        """Number of eigenfunctions per level"""
        raise NotImplementedError