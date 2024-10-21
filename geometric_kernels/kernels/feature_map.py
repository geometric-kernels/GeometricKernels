"""
This module provides the :class:`MaternFeatureMapKernel` kernel, the basic
kernel for non-compact symmetric spaces, subclasses of
:class:`~.spaces.NoncompactSymmetricSpace`.
"""

import lab as B
import numpy as np
from beartype.typing import Dict, Optional

from geometric_kernels.feature_maps import FeatureMap
from geometric_kernels.kernels.base import BaseGeometricKernel
from geometric_kernels.spaces.base import Space
from geometric_kernels.utils.utils import make_deterministic


class MaternFeatureMapKernel(BaseGeometricKernel):
    r"""
    This class computes a (Matérn) kernel based on a feature map.

    .. math :: k_{\nu, \kappa}(x, y) = \langle \phi_{\nu, \kappa}(x), \phi_{\nu, \kappa}(y) \rangle_{\mathbb{R}^n}

    where $\langle \cdot , \cdot \rangle_{\mathbb{R}^n}$ is the standard inner
    product in $\mathbb{R}^n$ and $\phi_{\nu, \kappa}: X \to \mathbb{R}^n$ is
    an arbitrary function called *feature map*. We assume that it depends
    on the smoothness and length scale parameters $\nu$ and $\kappa$,
    respectively, which makes this kernel specifically Matérn.

    .. note::
        A brief introduction into feature maps and related kernels can be
        found on :doc:`this page </theory/feature_maps>`.

        Note that the finite-dimensional feature maps this kernel is meant to
        be used with are, in most cases, some approximations of the
        intractable infinite-dimensional feature maps.

    :param space:
        The space on which the kernel is defined.
    :param feature_map:
        A :class:`~.feature_maps.FeatureMap` object that represents an
        arbitrary function $\phi_{\nu, \kappa}: X \to \mathbb{R}^n$, where
        $X$ is the `space`, $n$ can be an arbitrary finite integer, and
        $\nu, \kappa$ are the smoothness and length scale parameters.
    :param key:
        Random state, either `np.random.RandomState`,
        `tf.random.Generator`, `torch.Generator` or `jax.tensor` (which
        represents a random state).

        Many feature maps used in the library are randomized, thus requiring a
        `key` to work. The :class:`MaternFeatureMapKernel` uses this `key` to
        make them (and thus the kernel) deterministic, applying the utility
        function :func:`~.make_deterministic` to the pair `feature_map, key`.

        .. note::
           Even if the `feature_map` is deterministic, you need to provide a
           valid key, although it will essentially be ignored. In the future,
           we should probably make the `key` parameter optional.

    :param normalize:
        This parameter is directly passed on to the `feature_map` as a keyword
        argument "normalize". If normalize=True, then either $k(x, x) = 1$ for
        all $x \in X$, or $\int_X k(x, x) d x = 1$, depending on the type of
        the feature map and on the space $X$.

        .. note::
            For many kernel methods, $k(\cdot, \cdot)$ and $a k(\cdot, \cdot)$
            are indistinguishable, whatever the positive constant $a$ is. For
            these, it makes sense to use normalize=False to save up some
            computational overhead. For others, like for the Gaussian process
            regression, the normalization of the kernel might be important. In
            these cases, you will typically want to set normalize=True.
    """

    def __init__(
        self,
        space: Space,
        feature_map: FeatureMap,
        key: B.RandomState,
        normalize: bool = True,
    ):
        super().__init__(space)
        self.feature_map = make_deterministic(feature_map, key)
        self.normalize = normalize

    def init_params(self) -> Dict[str, B.NPNumeric]:
        """
        Initializes the dict of the trainable parameters of the kernel.

        Returns `dict(nu=np.array([np.inf]), lengthscale=np.array([1.0]))`.

        This dict can be modified and is passed around into such methods as
        :meth:`~.K` or :meth:`~.K_diag`, as the `params` argument.

        .. note::
            The values in the returned dict are always of the NumPy array type.
            Thus, if you want to use some other backend for internal
            computations when calling :meth:`~.K` or :meth:`~.K_diag`, you
            need to replace the values with the analogs typed as arrays of
            the desired backend.
        """
        params = dict(nu=np.array([np.inf]), lengthscale=np.array([1.0]))
        return params

    def K(
        self,
        params: Dict[str, B.Numeric],
        X: B.Numeric,
        X2: Optional[B.Numeric] = None,
        **kwargs,
    ):
        assert "lengthscale" in params
        assert params["lengthscale"].shape == (1,)
        assert "nu" in params
        assert params["nu"].shape == (1,)

        _, features_X = self.feature_map(
            X, params, normalize=self.normalize, **kwargs
        )  # [N, O]
        if X2 is not None:
            _, features_X2 = self.feature_map(
                X2, params, normalize=self.normalize, **kwargs
            )  # [N2, O]
        else:
            features_X2 = features_X

        feature_product = B.einsum("...no,...mo->...nm", features_X, features_X2)
        return feature_product

    def K_diag(self, params: Dict[str, B.Numeric], X: B.Numeric, **kwargs):
        assert "lengthscale" in params
        assert params["lengthscale"].shape == (1,)
        assert "nu" in params
        assert params["nu"].shape == (1,)

        _, features_X = self.feature_map(
            X, params, normalize=self.normalize, **kwargs
        )  # [N, O]
        return B.sum(features_X**2, axis=-1)  # [N, ]
