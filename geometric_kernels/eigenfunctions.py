"""
Eigenfunctions are callable objects which evaluate the eigenfunctions
of the Laplace-Beltrami operator on a manifold.
"""
import abc
from typing import Optional

import numpy as np
import tensorflow as tf

from geometric_kernels.types import TensorLike


class Eigenfunctions(abc.ABC):
    r"""
    Represents a set of eigenfunctions of an operator. Referred to as
    :math:`Phi = [\phi_i]_{i=0}^{M-1}`.
    """

    def weighted_outerproduct(
        self, weights: TensorLike, X: TensorLike, X2: Optional[TensorLike] = None
    ) -> TensorLike:
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

        weights = tf.reshape(weights, (-1,))
        return tf.einsum("ni,ki,i->nk", Phi_X, Phi_X2, weights)  # [N, N2]

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
        weights = tf.reshape(weights, (-1,))
        return tf.einsum("ni,i->n", Phi_X ** 2, weights)  # [N,]

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

    def weighted_outerproduct(
        self, weights: TensorLike, X: TensorLike, X2: Optional[TensorLike] = None
    ) -> TensorLike:
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
            sum_phi_phi_for_level = self._addition_theorem(X, X)  # [N, N, L]
            N1 = N2 = tf.shape(X)[0]
        else:
            sum_phi_phi_for_level = self._addition_theorem(X, X2)  # [N, N2, L]
            N1 = tf.shape(X)[0]
            N2 = tf.shape(X2)[0]

        weights = self._filter_weights(weights)
        weights = tf.reshape(weights, (-1,))  # flatten

        # shape checks
        tf.ensure_shape(sum_phi_phi_for_level, tf.TensorShape([N1, N2, self.num_levels]))
        tf.ensure_shape(
            weights,
            tf.TensorShape(
                [
                    self.num_levels,
                ]
            ),
        )

        return tf.einsum("i,nki->nk", weights, sum_phi_phi_for_level)  # [N, N2]

    def weighted_outerproduct_diag(self, weights: TensorLike, X: TensorLike) -> TensorLike:
        r"""
        Computes :math:`\sum_{i=0}^{M-1} w_i \phi_i(x) \phi_i(x)`. Corresponds to the
        diagonal elements of `weighted_outproduct` but they can be calculated more
        efficiently.

        Makes use of the fact that eigenfunctions within a level can be summed
        in a computationally more efficient matter.

        .. note:
            Only works if the weights within a level are equal.

        :param weights: [M, 1]
        :param X: Inputs where to evaluate the eigenfunctions, shape = [N, D].
        :return: shape [N,]
        """
        N = tf.shape(X)[0]
        addition_theorem_X = self._addition_theorem_diag(X)  # [N, L]

        weights = self._filter_weights(weights)
        weights = tf.reshape(weights, (-1,))  # flatten

        # shape checks
        tf.ensure_shape(addition_theorem_X, tf.TensorShape([N, self.num_levels]))
        tf.ensure_shape(
            weights,
            tf.TensorShape(
                [
                    self.num_levels,
                ]
            ),
        )
        return tf.einsum("i,ni->n", weights, addition_theorem_X)  # [N,]

    def _filter_weights(self, weights: TensorLike) -> TensorLike:
        """Selects the weight for each level"""
        weights_per_level = []
        # assumes the weights in `weights` within a level are the same
        # TODO(VD) write check for this.
        i = 0
        for num in self.num_eigenfunctions_per_level:
            weights_per_level.append(weights[i])
            i += num
        return tf.reshape(tf.convert_to_tensor(weights_per_level), (-1, 1))

    @abc.abstractmethod
    def _addition_theorem(self, X: TensorLike, X2: TensorLike) -> TensorLike:
        """
        Returns the sum of eigenfunctions on a level for which we have a simplified expression

        :param X: [N, D]
        :param X2: [N2, D]
        :return: Evaluate the sum of eigenfunctions on each level. Returns
            a value for each level [N, N2, L]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _addition_theorem_diag(self, X: TensorLike) -> TensorLike:
        """
        Returns the sum of eigenfunctions on a level for which we have a simplified expression

        :param X: [N, D]
        :return: Evaluate the sum of eigenfunctions on each level. Returns
            a value for each level [N, L]
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
