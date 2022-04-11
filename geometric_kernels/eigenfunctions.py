"""
Eigenfunctions are callable objects which evaluate the eigenfunctions
of the Laplace-Beltrami operator on a manifold.
"""
import abc

import lab as B
from opt_einsum import contract as einsum

from geometric_kernels.lab_extras.extras import from_numpy
from geometric_kernels.utils import Optional


class Eigenfunctions(abc.ABC):
    r"""
    Represents a set of eigenfunctions of an operator. Referred to as
    :math:`Phi = [\phi_i]_{i=0}^{M-1}`.
    """

    def weighted_outerproduct(
        self,
        weights: B.Numeric,
        X: B.Numeric,
        X2: Optional[B.Numeric] = None,  # type: ignore
        **parameters,
    ) -> B.Numeric:
        r"""
        Computes :math:`\sum_{i=0}^{M-1} w_i \phi_i(x1) \phi_i(x2)`.

        :param weights: [M, 1]
        :param X: Inputs where to evaluate the eigenfunctions, shape = [N, D]
            where `N` is the number of inputs and `D` the dimension.
        :param X2: Inputs where to evaluate the eigenfunctions, shape = [N2, D],
            where `N` is the number of inputs and `D` the dimension.
            Default to None, in which X is used for X2.
        :param parameters: any additional parameters
        :return: shape [N, N2]
        """
        Phi_X = self.__call__(X, **parameters)  # [N, L]
        if X2 is None:
            Phi_X2 = Phi_X
        else:
            Phi_X2 = self.__call__(X2, **parameters)  # [N2, L]

        Phi_X = B.cast(B.dtype(weights), Phi_X)
        Phi_X2 = B.cast(B.dtype(weights), Phi_X2)

        Kxx = B.matmul(B.transpose(weights) * Phi_X, Phi_X2, tr_b=True)  # [N, N2]
        return Kxx

    def weighted_outerproduct_diag(
        self, weights: B.Numeric, X: B.Numeric, **parameters
    ) -> B.Numeric:
        r"""
        Computes :math:`\sum_{i=0}^{M-1} w_i \phi_i(x) \phi_i(x)`. Corresponds to the
        diagonal elements of `weighted_outproduct` but they can be calculated more
        efficiently.

        :param weights: [M, 1]
        :param X: Inputs where to evaluate the eigenfunctions, shape = [N, D].
        :param parameters: any additional parameters
        :return: shape [N,]
        """
        Phi_X = self.__call__(X, **parameters)  # [N, L]
        Kx = B.sum(B.transpose(weights) * Phi_X**2, axis=1)  # [N,]
        return Kx

    @abc.abstractmethod
    def __call__(self, X: B.Numeric, **parameters) -> B.Numeric:
        """
        :param X: points to evaluate the eigenfunctions in local coordinates, [N, D].
            `N` is the number of points and `D` should match the dimension of the space
            on which the eigenfunctions are defined.
        :param parameters: any additional parameters
        """
        raise NotImplementedError

    @abc.abstractproperty
    def num_eigenfunctions(self) -> int:
        """Number of eigenfunctions, M"""
        raise NotImplementedError


class EigenfunctionWithAdditionTheorem(Eigenfunctions):
    r"""
    Eigenfunctions for which the sum over a level has a simpler expression.

    Example 1:
    On the circle S^1 the eigenfunctions are given by :math:`{\sin(l \theta), \cos(l \theta)}`,
    where we refer to :math:`l` as the level. Summing over the eigenfunctions of a level
    as follows :math:`\cos(l x) \cos(l x') + \sin(l x) \sin(l x)` can be simplified to
    :math:`cos(l (x-x'))` thanks to some trigonometric identity.

    Example 2:
    The sphere manifold S^d eigenfunctions, known as the spherical harmonics, also adhere
    to this property. It is known as the addition theorem.  See, for example, Theorem 4.11 (p.60
     Frye and Efthimiou (2012).

    In the case the weights over a level in the `weighted_outproduct` are identical
    we can make use of this expression to simplify computations.

    We assume there are `L` levels. The sum of the number of eigenfunctions per level should be
    equal the total amount of eigenfunctions.
    """

    def weighted_outerproduct(
        self,
        weights: B.Numeric,
        X: B.Numeric,
        X2: Optional[B.Numeric] = None,  # type: ignore
        **parameters,
    ) -> B.Numeric:
        r"""
        Computes :math:`\sum w_i \phi_i(x1) \phi_i(x2)`.

        :param weights: [L, 1]
            .. note:
                The length of `weights` is equal to the number of levels.
                This is **not** the same as the number of eigenfunctions.

        :param X: Inputs where to evaluate the eigenfunctions, shape = [N, D].
        :param X2: Inputs where to evaluate the eigenfunctions, shape = [N2, D].
            Default to None, in which X is used for X2.
        :param parameters: any additional parameters
        :return: shape [N, N2]
        """
        if X2 is None:
            X2 = X

        sum_phi_phi_for_level = self._addition_theorem(X, X2, **parameters)  # [N, N, L]
        weights = self._filter_weights(weights)
        sum_phi_phi_for_level = B.cast(B.dtype(weights), sum_phi_phi_for_level)

        return einsum("i,nki->nk", weights, sum_phi_phi_for_level)  # [N, N2]

    def weighted_outerproduct_diag(
        self, weights: B.Numeric, X: B.Numeric, **parameters
    ) -> B.Numeric:
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
        :param parameters: any additional parameters
        :return: shape [N,]
        """
        addition_theorem_X = self._addition_theorem_diag(X, **parameters)  # [N, L]
        weights = self._filter_weights(weights)
        # weights = from_numpy(addition_theorem_X, weights)
        return einsum("i,ni->n", weights, addition_theorem_X)  # [N,]

    def _filter_weights(self, weights: B.Numeric) -> B.Numeric:
        """
        Selects the weight for each level.
        Assumes the weights in `weights` within a level are the same.

        :param weights: [M,]
        :return: [L,]
        """
        weights_per_level = []
        # assumes the weights in `weights` within a level are the same
        # TODO(VD) write check for this.
        i = 0
        for num in self.num_eigenfunctions_per_level:
            weights_per_level.append(weights[i] * B.cast(B.dtype(weights), B.ones(1)))
            i += num
        return B.concat(*weights_per_level, axis=0)  # [L,]

    @abc.abstractmethod
    def _addition_theorem(self, X: B.Numeric, X2: B.Numeric, **parameters) -> B.Numeric:
        """
        Returns the sum of eigenfunctions on a level for which we have a simplified expression

        :param X: [N, D]
        :param X2: [N2, D]
        :param parameters: any additional parameters
        :return: Evaluate the sum of eigenfunctions on each level. Returns
            a value for each level [N, N2, L]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _addition_theorem_diag(self, X: B.Numeric, **parameters) -> B.Numeric:
        """
        Returns the sum of eigenfunctions on a level for which we have a simplified expression

        :param X: [N, D]
        :param parameters: any additional parameters
        :return: Evaluate the sum of eigenfunctions on each level. Returns
            a value for each level [N, L]
        """
        raise NotImplementedError

    @abc.abstractproperty
    def num_levels(self) -> int:
        """Number of levels, L"""
        raise NotImplementedError

    @abc.abstractproperty
    def num_eigenfunctions_per_level(self) -> B.Numeric:
        """Number of eigenfunctions per level"""
        raise NotImplementedError
