"""
This module provides base classes for storing or evaluating eigenfunctions of
the Laplacian, or certain combinations thereof (see the note below).

.. note::
    Sometimes, when relations analogous to the :doc:`addition theorem
    </theory/addition_theorem>` on the sphere are available, it is much more
    efficient to use certain sums of outer products of eigenfunctions instead
    of the eigenfunctions themselves. For this, we offer
    :class:`EigenfunctionsWithAdditionTheorem`. Importantly, it is permitted to
    _only_ provide the computational routines for these "certain sums", lacking
    the actual capability to compute the eigenfunctions themselves. This is
    important because for compact Lie groups, for example, computing
    eigenfunctions is more involved and less efficient than computing the
    "certain sums", called characters in this case. This is also relevant for
    hyperspheres of higher dimension: in this case, the eigenfunctions
    (spherical harmonics) are much more cumbersome than the "certain
    sums" (zonal spherical harmonics).
"""

import abc

import lab as B
from beartype.typing import List, Optional

from geometric_kernels.lab_extras import complex_like, is_complex, take_along_axis


class Eigenfunctions(abc.ABC):
    r"""
    Abstract base class providing an interface for working with eigenfunctions.

    We denote the basis of eigenfunctions represented by an instance of
    this class by $[f_j]_{j=0}^{J-1}$. We assume it to be partitioned into
    *levels* (see :class:`~.kernels.MaternKarhunenLoeveKernel` on how they are
    used). Specifically, we call the sets $[f_{l s}]_{s=1}^{d_l}$ *levels* if

    .. math:: [f_j]_{j=0}^{J-1} = \bigcup_{l=0}^{L-1}[f_{l s}]_{s=1}^{d_l}

    such that all the eigenfunctions $f_{l s}$ with the same index $l$
    correspond to the same eigenvalue $\lambda_l$. Note, however, that
    $\lambda_l$ are not required to be unique: it is possible that for some
    $l,l'$, $\lambda_l = \lambda_{l'}$.

    .. note::
        There is often more than one way to choose a partition into levels.
        Trivially, you can always correspond a level to each individual
        eigenfunction. Alternatively, you can partition $[f_j]_{j=0}^{J-1}$
        into the maximal subsets corresponding to the same eigenvalue (into
        the *full eigenspaces*). There are also often plenty of possibilities
        in between these two extremes.

    Importantly, subclasses of this class do not necessarily have to allow
    computing the individual eigenfunctions (i.e. implement the method
    :meth:`__call__`). The only methods that subclasses *have to* implement
    are :meth:`phi_product` and its close relative :meth:`phi_product_diag`.
    These output the sums of outer products of eigenfunctions for all levels:

    .. math:: \sum_{s=1}^{d_l} f_{l s}(x_1) f_{l s}(x_2)

    for all $0 \leq l < L$, and all pairs $x_1$, $x_2$ provided as inputs.
    """

    def weighted_outerproduct(
        self,
        weights: B.Numeric,
        X: B.Numeric,
        X2: Optional[B.Numeric] = None,  # type: ignore
        **kwargs,
    ) -> B.Numeric:
        r"""
        Computes

        .. math:: \sum_{l=0}^{L-1} w_l \sum_{s=1}^{d_l} f_{l s}(x_1) f_{l s}(x_2).

        for all $x_1$ in `X` and all $x_2$ in `X2`, where $w_l$ are `weights`.

        :param weights:
            An array of shape [L, 1] where L is the number of levels.
        :param X:
            The first of the two batches of points to evaluate the weighted
            outer product at. An array of shape [N, <axis>], where N is the
            number of points and <axis> is the shape of the arrays that
            represent the points in a given space.
        :param X2:
            The second of the two batches of points to evaluate the weighted
            outer product at. An array of shape [N2, <axis>], where N2 is the
            number of points and <axis> is the shape of the arrays that
            represent the points in a given space.

            Defaults to None, in which case X is used for X2.
        :param ``**kwargs``:
            Any additional parameters.

        :return:
            An array of shape [N, N2].
        """
        if X2 is None:
            X2 = X

        sum_phi_phi_for_level = self.phi_product(X, X2, **kwargs)  # [N, N2, L]

        if is_complex(sum_phi_phi_for_level):
            sum_phi_phi_for_level = B.cast(complex_like(weights), sum_phi_phi_for_level)
            weights = B.cast(complex_like(weights), weights)
        else:
            sum_phi_phi_for_level = B.cast(B.dtype(weights), sum_phi_phi_for_level)

        return B.einsum("id,...nki->...nk", weights, sum_phi_phi_for_level)  # [N, N2]

    def weighted_outerproduct_diag(
        self, weights: B.Numeric, X: B.Numeric, **kwargs
    ) -> B.Numeric:
        r"""
        Computes the diagonal of the matrix
        ``weighted_outerproduct(weights, X, X, **kwargs)``.

        :param weights:
            As in :meth:`weighted_outerproduct`.
        :param X:
            As in :meth:`weighted_outerproduct`.
        :param ``**kwargs``:
            As in :meth:`weighted_outerproduct`.

        :return:
            An array of shape [N,].
        """
        phi_product_diag = self.phi_product_diag(X, **kwargs)  # [N, L]

        if is_complex(phi_product_diag):
            phi_product_diag = B.cast(complex_like(weights), phi_product_diag)
            weights = B.cast(complex_like(weights), weights)
        else:
            phi_product_diag = B.cast(B.dtype(weights), phi_product_diag)

        return B.einsum("id,ni->n", weights, phi_product_diag)  # [N,]

    @abc.abstractmethod
    def phi_product(
        self, X: B.Numeric, X2: Optional[B.Numeric] = None, **kwargs
    ) -> B.Numeric:
        r"""
        Computes the

        .. math:: \sum_{s=1}^{d_l} f_{l s}(x_1) f_{l s}(x_2)

        for all $x_1$ in `X`, all $x_2$ in `X2`, and $0 \leq l < L$.

        :param X:
            The first of the two batches of points to evaluate the phi
            product at. An array of shape [N, <axis>], where N is the
            number of points and <axis> is the shape of the arrays that
            represent the points in a given space.
        :param X2:
            The second of the two batches of points to evaluate the phi
            product at. An array of shape [N2, <axis>], where N2 is the
            number of points and <axis> is the shape of the arrays that
            represent the points in a given space.

            Defaults to None, in which case X is used for X2.
        :param ``**kwargs``:
            Any additional parameters.

        :return:
            An array of shape [N, N2, L].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def phi_product_diag(self, X: B.Numeric, **kwargs):
        r"""
        Computes the diagonals of the matrices
        ``phi_product(X, X, **kwargs)[:, :, l]``, $0 \leq l < L$.

        :param X:
            As in :meth:`phi_product`.
        :param ``**kwargs``:
            As in :meth:`phi_product`.

        :return:
            An array of shape [N, L].
        """
        raise NotImplementedError

    def __call__(self, X: B.Numeric, **kwargs) -> B.Numeric:
        """
        Evaluate the individual eigenfunctions at a batch of input locations.

        :param X:
            Points to evaluate the eigenfunctions at, an array of
            shape [N, <axis>], where N is the number of points and <axis> is
            the shape of the arrays that represent the points in a given space.
        :param ``**kwargs``:
            Any additional parameters.

        :return:
            An [N, J]-shaped array, where `J` is the number of eigenfunctions.
        """
        raise NotImplementedError

    @property
    def num_eigenfunctions(self) -> int:
        """
        The number J of eigenfunctions.
        """
        return sum(self.num_eigenfunctions_per_level)

    @property
    def num_levels(self) -> int:
        """
        The number L of levels.
        """
        return len(self.num_eigenfunctions_per_level)

    @abc.abstractproperty
    def num_eigenfunctions_per_level(self) -> List[int]:
        r"""
        The number of eigenfunctions per level: list of $d_l$, $0 \leq l < L$.
        """
        raise NotImplementedError


class EigenfunctionsWithAdditionTheorem(Eigenfunctions):
    r"""
    Eigenfunctions for which the sum of outer products over a level has a
    simpler expression.

    .. note::
        See a brief introduction into the notion of addition theorems in the
        :doc:`respective documentation page </theory/addition_theorem>`.
    """

    def phi_product(
        self, X: B.Numeric, X2: Optional[B.Numeric] = None, **kwargs
    ) -> B.Numeric:
        return self._addition_theorem(X, X2, **kwargs)

    def phi_product_diag(self, X: B.Numeric, **kwargs) -> B.Numeric:
        return self._addition_theorem_diag(X, **kwargs)

    @abc.abstractmethod
    def _addition_theorem(
        self, X: B.Numeric, X2: Optional[B.Numeric] = None, **kwargs
    ) -> B.Numeric:
        """
        Basically, an implementation of :meth:`phi_product`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _addition_theorem_diag(self, X: B.Numeric, **kwargs) -> B.Numeric:
        """
        Basically, an implementation of :meth:`phi_product_diag`.
        """
        raise NotImplementedError


class EigenfunctionsFromEigenvectors(Eigenfunctions):
    """
    Turns an array of eigenvectors into an :class:`Eigenfunctions` instance.
    The resulting eigenfunctions' inputs are given by the indices.

    Each individual eigenfunction corresponds to a separate level.

    :param eigenvectors:
        Array of shape [D, J] containing J eigenvectors of dimension D.
    """

    def __init__(self, eigenvectors: B.Numeric):
        self.eigenvectors = eigenvectors

    def weighted_outerproduct(
        self,
        weights: B.Numeric,
        X: B.Numeric,
        X2: Optional[B.Numeric] = None,  # type: ignore
        **kwargs,
    ) -> B.Numeric:
        Phi_X = self.__call__(X, **kwargs)  # [N, J]
        if X2 is None:
            Phi_X2 = Phi_X
        else:
            Phi_X2 = self.__call__(X2, **kwargs)  # [N2, J]

        Phi_X = B.cast(B.dtype(weights), Phi_X)
        Phi_X2 = B.cast(B.dtype(weights), Phi_X2)

        Kxx = B.matmul(B.transpose(weights) * Phi_X, Phi_X2, tr_b=True)  # [N, N2]
        return Kxx

    def weighted_outerproduct_diag(
        self, weights: B.Numeric, X: B.Numeric, **kwargs
    ) -> B.Numeric:
        Phi_X = self.__call__(X, **kwargs)  # [N, J]
        Kx = B.sum(B.transpose(weights) * Phi_X**2, axis=1)  # [N,]
        return Kx

    def phi_product(
        self, X: B.Numeric, X2: Optional[B.Numeric] = None, **kwargs
    ) -> B.Numeric:
        if X2 is None:
            X2 = X
        Phi_X = self.__call__(X, **kwargs)  # [N, J]
        Phi_X2 = self.__call__(X2, **kwargs)  # [N2, J]
        return B.einsum("nl,ml->nml", Phi_X, Phi_X2)  # [N, N2, J]

    def phi_product_diag(self, X: B.Numeric, **kwargs):
        Phi_X = self.__call__(X, **kwargs)  # [N, J]
        return Phi_X**2

    def __call__(self, X: B.Numeric, **kwargs) -> B.Numeric:
        """
        Takes the values of the `J` stored eigenvectors `self.eigenvectors`
        that correspond to the indices in `X`.

        :param X:
            Indices, an array of shape [N, 1].
        :param ``**kwargs``:
            Ignored.

        :return:
            An array of shape [N, J], whose element with index (n, j)
            corresponds to the X[n]-th element of the j-th eigenvector.
        """
        indices = B.cast(B.dtype_int(X), X)
        Phi = take_along_axis(self.eigenvectors, indices, axis=0)
        return Phi

    @property
    def num_eigenfunctions(self) -> int:
        return B.shape(self.eigenvectors)[-1]

    @property
    def num_levels(self) -> int:
        """
        Number of levels.

        Returns J, same as :meth:`num_eigenfunctions`.
        """
        return self.num_eigenfunctions

    @property
    def num_eigenfunctions_per_level(self) -> List[int]:
        """
        Number of eigenfunctions per level.

        Returns a list of J ones.
        """
        return [1] * self.num_levels
