"""
This module provides the :class:`BaseGeometricKernel` kernel, the base class
for all geometric kernels.
"""

import abc

import lab as B
from beartype.typing import Dict, List, Optional, Union

from geometric_kernels.spaces import Space


class BaseGeometricKernel(abc.ABC):
    """
    Abstract base class for geometric kernels.

    :param space:
        The space on which the kernel is defined.
    """

    def __init__(self, space: Space):
        self._space = space

    @property
    def space(self) -> Union[Space, List[Space]]:
        """
        The space on which the kernel is defined.
        """
        return self._space

    @abc.abstractmethod
    def init_params(self) -> Dict[str, B.NPNumeric]:
        """
        Initializes the dict of the trainable parameters of the kernel.

        It typically contains only two keys: `"nu"` and `"lengthscale"`.

        This dict can be modified and is passed around into such methods as
        :meth:`~.K` or :meth:`~.K_diag`, as the `params` argument.

        .. note::
            The values in the returned dict are always of the NumPy array type.
            Thus, if you want to use some other backend for internal
            computations when calling :meth:`~.K` or :meth:`~.K_diag`, you
            need to replace the values with the analogs typed as arrays of
            the desired backend.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def K(
        self,
        params: Dict[str, B.Numeric],
        X: B.Numeric,
        X2: Optional[B.Numeric] = None,
        **kwargs,
    ) -> B.Numeric:
        """
        Compute the cross-covariance matrix between two batches of vectors of
        inputs, or batches of matrices of inputs, depending on the space.

        :param params:
            A dict of kernel parameters, typically containing two keys:
            `"lengthscale"` for length scale and `"nu"` for smoothness.

            The types of values in the params dict determine the output type
            and the backend used for the internal computations, see the
            warning below for more details.

            .. note::
                The values `params["lengthscale"]` and `params["nu"]` are
                typically (1,)-shaped arrays of the suitable backend. This
                serves to point at the backend to be used for internal
                computations.

                In some cases, for example, when the kernel is
                :class:`~.kernels.ProductGeometricKernel`, the values of
                `params` may be (s,)-shaped arrays instead, where `s` is the
                number of factors.

            .. note::
                Finite values of `params["nu"]` typically correspond to the
                generalized (geometric) Matérn kernels.

                Infinite `params["nu"]` typically corresponds to the heat
                kernel (a.k.a. diffusion kernel, generalized squared
                exponential kernel, generalized Gaussian kernel,
                generalized RBF kernel). Although it is often considered to be
                a separate entity, we treat the heat kernel as a member of
                the Matérn family, with smoothness parameter equal to infinity.

        :param X:
            A batch of N inputs, each of which is a vector or a matrix,
            depending on how the elements of the `self.space` are represented.
        :param X2:
            A batch of M inputs, each of which is a vector or a matrix,
            depending on how the elements of the `self.space` are represented.

            `X2=None` sets `X2=X1`.

            Defaults to None.

        :return:
            The N x M cross-covariance matrix.

        .. warning::
           The types of values in the `params` dict determine the backend
           used for internal computations and the output type.

           Even if, say, `geometric_kernels.jax` is imported but the values in
           the `params` dict are NumPy arrays, the output type will be a NumPy
           array, and NumPy will be used for internal computations. To get a
           JAX array as an output and use JAX for internal computations, all
           the values in the `params` dict must be JAX arrays.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def K_diag(self, params: Dict[str, B.Numeric], X: B.Numeric, **kwargs) -> B.Numeric:
        """
        Returns the diagonal of the covariance matrix `self.K(params, X, X)`,
        typically in a more efficient way than actually computing the full
        covariance matrix with `self.K(params, X, X)` and then extracting its
        diagonal.

        :param params:
            Same as for :meth:`~.K`.

        :param X:
            A batch of N inputs, each of which is a vector or a matrix,
            depending on how the elements of the `self.space` are represented.

        :return:
            The N-dimensional vector representing the diagonal of the
            covariance matrix `self.K(params, X, X)`.
        """
        raise NotImplementedError
