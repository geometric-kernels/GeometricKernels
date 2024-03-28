"""
This module provides the :class:`MaternFeatureMapKernel` kernel, the basic
kernel for non-compact symmetric spaces, subclasses of :class:`NoncompactSymmetricSpace`.
"""
import lab as B
import numpy as np
from opt_einsum import contract as einsum

from geometric_kernels.kernels.base import BaseGeometricKernel
from geometric_kernels.spaces.base import Space
from geometric_kernels.utils.utils import make_deterministic


class MaternFeatureMapKernel(BaseGeometricKernel):
    r"""
    This class computes a (Matérn) kernel based on a feature map.

    For every kernel `k` on a space `X`, there is a map :math:`\phi` from the space `X`
    to some (possibly infinite-dimensional) space :math:`\mathcal{H}` such that:

    .. math :: k(x, y) = \langle \phi(x), \phi(y) \rangle_{\mathcal{H}}

    where :math:`\langle \cdot , \cdot \rangle_{\mathcal{H}}` means inner
    product in :math:`\mathcal{H}`.

    One can approximate the kernel using a finite-dimensional approximation to
    :math:`\phi` which we call a `feature map`.

    What makes this kernel specifically Matérn is that it has
    a smoothness parameter `nu` and a lengthscale parameter `lengthscale`.

    .. note::
        A brief introduction into feature maps and related kernels can be found
        on :doc:`this page </theory/feature_maps>`.
    """

    def __init__(self, space: Space, feature_map, key, normalize=True):
        super().__init__(space)
        self.feature_map = make_deterministic(feature_map, key)
        self.normalize = normalize

    def init_params(self):
        params = dict(nu=np.array(np.inf), lengthscale=np.array(1.0))
        return params

    def K(self, params, X, X2=None, **kwargs):
        _, features_X = self.feature_map(
            X, params, normalize=self.normalize, **kwargs
        )  # [N, O]
        if X2 is not None:
            _, features_X2 = self.feature_map(
                X2, params, normalize=self.normalize, **kwargs
            )  # [M, O]
        else:
            features_X2 = features_X

        feature_product = einsum("...no,...mo->...nm", features_X, features_X2)
        return feature_product

    def K_diag(self, params, X, **kwargs):
        _, features_X = self.feature_map(
            X, params, normalize=self.normalize, **kwargs
        )  # [N, O]
        return B.sum(features_X**2, axis=-1)  # [N, ]
